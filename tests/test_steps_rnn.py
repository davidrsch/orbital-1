"""Tests for rnn pipeline step translators."""

import math

import onnx
import ibis
import numpy as np
import pytest
from onnx import TensorProto, helper

from orbital.translate import TRANSLATORS
from orbital.translation.variables import (
    GraphVariables,
    NumericVariablesGroup,
    ValueVariablesGroup,
    VariablesGroup,
)
from orbital.translation.optimizer import Optimizer
from orbital.translation.options import TranslationOptions
from orbital.translation.steps.softmax import SoftmaxTranslator
from orbital.translation.steps.imputer import ImputerTranslator
from orbital.translation.steps.argmax import ArgMaxTranslator
from orbital.translation.steps.add import AddTranslator
from orbital.translation.steps.sub import SubTranslator
from orbital.translation.steps.mul import MulTranslator
from orbital.translation.steps.div import DivTranslator
from orbital.translation.steps.identity import IdentityTranslator
from orbital.translation.steps.reshape import ReshapeTranslator
from orbital.translation.steps.matmul import MatMulTranslator
from orbital.translation.steps.cast import CastTranslator, CastLikeTranslator
from orbital.translation.steps.linearclass import LinearClassifierTranslator
from orbital.translation.steps.linearreg import LinearRegressorTranslator
from orbital.translation.steps.scaler import ScalerTranslator
from orbital.translation.steps.onehotencoder import OneHotEncoderTranslator
from orbital.translation.steps.labelencoder import LabelEncoderTranslator
from orbital.translation.steps.where import WhereTranslator
from orbital.translation.steps.zipmap import ZipMapTranslator
from orbital.translation.steps.concat import ConcatTranslator
from orbital.translation.steps.featurevectorizer import FeatureVectorizerTranslator
from orbital.translation.steps.gather import GatherTranslator
from orbital.translation.steps.arrayfeatureextractor import ArrayFeatureExtractorTranslator



# ---------------------------------------------------------------------------
# Helper: build an ONNX graph with weight initializers
# ---------------------------------------------------------------------------

def _make_graph_with_inits(node, inputs_info, outputs_info, initializers):
    """Create an ONNX GraphProto with given node, I/O specs, and initializers."""
    return helper.make_graph(
        [node],
        "test_graph",
        inputs_info,
        outputs_info,
        initializer=initializers,
    )

# ---------------------------------------------------------------------------
# Unit tests: LSTMTranslator
# ---------------------------------------------------------------------------


class TestLSTMTranslator:
    """Tests for the ONNX LSTM translator (forward-direction unrolling)."""


    def _make_lstm_graph(self, I, H, T, W_data, R_data, B_data=None):
        """Build a minimal ONNX LSTM graph with Y_h output."""
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, 4 * H, I], W_data)
        R_tensor = helper.make_tensor("R", TensorProto.FLOAT, [1, 4 * H, H], R_data)
        inputs = ["X", "W", "R"]
        inits = [W_tensor, R_tensor]
        if B_data is not None:
            B_tensor = helper.make_tensor("B", TensorProto.FLOAT, [1, 8 * H], B_data)
            inputs.append("B")
            inits.append(B_tensor)
        node = helper.make_node(
            "LSTM",
            inputs=inputs,
            outputs=["", "Y_h"],
            hidden_size=H,
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I])],
            [helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [None, H])],
            inits,
        )
        return graph

    def test_lstm_registered(self):
        from orbital.translation.steps.lstm import LSTMTranslator
        assert TRANSLATORS.get("LSTM") is LSTMTranslator

    def test_lstm_single_step_zero_weights(self):
        """Single timestep, all-zero weights, no bias.

        All gates  0.5 (sigmoid(0)) or 0.0 (tanh(0)).
        i=0.5, f=0.5, c_bar=0.0, o=0.5  C=0.5*0.0, H=0.5*tanh(0)=0
        """
        from orbital.translation.steps.lstm import LSTMTranslator

        I, H, T = 2, 2, 1
        W_data = [0.0] * (4 * H * I)  # all zeros
        R_data = [0.0] * (4 * H * H)

        graph = self._make_lstm_graph(I, H, T, W_data, R_data)
        table = ibis.memtable({"x0": [0.5], "x1": [-0.5]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"], "x1": table["x1"]})

        LSTMTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y_h")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == H

        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        for v in vals:
            assert abs(v) < 1e-9  # H = 0.5 * tanh(0) = 0

    def test_lstm_identity_forward(self):
        """T=2 timesteps, identity-like weights, no bias, both outputs approx equal."""
        from orbital.translation.steps.lstm import LSTMTranslator

        I, H, T = 1, 1, 2
        # W[gate, unit, input]: set only C-gate (gate 3 in IOFC order = index 3)
        # to 1.0 so c_bar captures the input.
        W_data = [0.0] * (4 * H * I)
        W_data[3 * H * I] = 1.0   # W_c[0, 0] = 1.0

        R_data = [0.0] * (4 * H * H)
        # i-gate W = const 3 (large sigmoid  ~1.0), f-gate W = 0
        W_data[0 * H * I] = 10.0  # W_i[0, 0]: large positive  i1

        graph = self._make_lstm_graph(I, H, T, W_data, R_data)
        table = ibis.memtable({"x0_t0": [1.0], "x0_t1": [0.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup(
            {"x0_t0": table["x0_t0"], "x0_t1": table["x0_t1"]}
        )

        LSTMTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y_h")
        assert result is not None
        assert len(result) == H

    def test_lstm_rejects_bidirectional(self):
        """Bidirectional LSTM with 1-direction weights raises ValueError."""
        from orbital.translation.steps.lstm import LSTMTranslator

        I, H, T = 1, 1, 1
        W_data = [0.0] * (4 * H * I)
        R_data = [0.0] * (4 * H * H)
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, 4 * H, I], W_data)
        R_tensor = helper.make_tensor("R", TensorProto.FLOAT, [1, 4 * H, H], R_data)

        node = helper.make_node(
            "LSTM",
            inputs=["X", "W", "R"],
            outputs=["", "Y_h"],
            hidden_size=H,
            direction="bidirectional",
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I])],
            [helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [None, H])],
            [W_tensor, R_tensor],
        )
        table = ibis.memtable({"x0": [1.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"]})

        with pytest.raises(ValueError, match="direction"):
            LSTMTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()

    def test_lstm_sequence_lens_raises_not_implemented(self):
        """sequence_lens (input[4]) must raise NotImplementedError, not a
        misleading peephole error.

        Updated from the earlier regression test that expected no raise:
        sequence_lens is now explicitly unsupported (M-8) so that callers
        get a clear error instead of silent wrong-answer behaviour.
        """
        from orbital.translation.steps.lstm import LSTMTranslator

        I, H, T = 1, 1, 1
        W_data = [0.0] * (4 * H * I)
        R_data = [0.0] * (4 * H * H)
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, 4 * H, I], W_data)
        R_tensor = helper.make_tensor("R", TensorProto.FLOAT, [1, 4 * H, H], R_data)
        # B, sequence_lens, initial_h, initial_c are all empty/absent; no P.
        node = helper.make_node(
            "LSTM",
            inputs=["X", "W", "R", "", "seq_lens"],  # sequence_lens present, no P
            outputs=["", "Y_h"],
            hidden_size=H,
        )
        seq_lens_tensor = helper.make_tensor("seq_lens", TensorProto.INT32, [1], [1])
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I])],
            [helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [None, H])],
            [W_tensor, R_tensor, seq_lens_tensor],
        )
        table = ibis.memtable({"x0": [1.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"]})

        # Must raise NotImplementedError — sequence_lens is unsupported (M-8).
        with pytest.raises(NotImplementedError, match="sequence_lens"):
            LSTMTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()

    def test_lstm_gate_order_iofc_regression(self):
        """Gate-order regression: verifies ONNX IOFC gate ordering is applied.

        Arrangement: only W[gate=3 (C), unit=0, input=0] = 1.0 and
        W[gate=0 (I), unit=0, input=0] = 100.0 (drives i_gate ≈ 1.0).
        All R = 0, no bias,  T=1, I=1, H=1,  x = [1.0].

        Expected under IOFC:
          i_gate  = sigmoid(100)  ≈ 1.0
          o_gate  = sigmoid(0)    = 0.5
          f_gate  = sigmoid(0)    = 0.5
          c_bar   = tanh(1.0)     ≈ 0.7616
          C_new   = 0.5*0 + 1.0*tanh(1.0) = tanh(1.0)
          H_out   = 0.5 * tanh(tanh(1.0)) ≈ 0.3211

        A different gate ordering (e.g. IFCO) would map W[3] to the O-gate
        instead of C-gate, yielding H_out = o*tanh(C) = sigmoid(1)*tanh(0) = 0.
        """
        import math
        from orbital.translation.steps.lstm import LSTMTranslator

        I, H, T = 1, 1, 1
        W_data = [0.0] * (4 * H * I)
        W_data[0 * H * I] = 100.0   # gate 0 = I-gate: force i ≈ 1.0
        W_data[3 * H * I] = 1.0     # gate 3 = C-gate: c_bar = tanh(x)
        R_data = [0.0] * (4 * H * H)

        graph = self._make_lstm_graph(I, H, T, W_data, R_data)
        table = ibis.memtable({"x0": [1.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"]})

        LSTMTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y_h")
        assert isinstance(result, ValueVariablesGroup)
        backend = ibis.duckdb.connect()
        h_out = backend.execute(list(result.values())[0]).tolist()[0]

        # Expected: 0.5 * tanh(tanh(1.0))
        expected = 0.5 * math.tanh(math.tanh(1.0))
        assert abs(h_out - expected) < 1e-6, (
            f"LSTM gate-order regression failed: got {h_out}, expected {expected}. "
            "If this fails, gate index 3 is not being used as the C-gate (IOFC ordering violated)."
        )


# ---------------------------------------------------------------------------
# Unit tests: GRUTranslator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit tests: GRUTranslator
# ---------------------------------------------------------------------------


class TestGRUTranslator:
    """Tests for the ONNX GRU translator (forward-direction unrolling)."""


    def _make_gru_graph(self, I, H, T, W_data, R_data, B_data=None):
        """Build a minimal ONNX GRU graph with Y_h output."""
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, 3 * H, I], W_data)
        R_tensor = helper.make_tensor("R", TensorProto.FLOAT, [1, 3 * H, H], R_data)
        inputs = ["X", "W", "R"]
        inits = [W_tensor, R_tensor]
        if B_data is not None:
            B_tensor = helper.make_tensor("B", TensorProto.FLOAT, [1, 6 * H], B_data)
            inputs.append("B")
            inits.append(B_tensor)
        node = helper.make_node(
            "GRU",
            inputs=inputs,
            outputs=["", "Y_h"],
            hidden_size=H,
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I])],
            [helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [None, H])],
            inits,
        )
        return graph

    def test_gru_registered(self):
        from orbital.translation.steps.gru import GRUTranslator
        assert TRANSLATORS.get("GRU") is GRUTranslator

    def test_gru_single_step_zero_weights(self):
        """All-zero weights: z0.5, r0.5, h=0  H=(1-0.5)*0+0.5*0=0."""
        from orbital.translation.steps.gru import GRUTranslator

        I, H, T = 2, 2, 1
        W_data = [0.0] * (3 * H * I)
        R_data = [0.0] * (3 * H * H)

        graph = self._make_gru_graph(I, H, T, W_data, R_data)
        table = ibis.memtable({"x0": [0.5], "x1": [-0.5]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"], "x1": table["x1"]})

        GRUTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y_h")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == H

        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        for v in vals:
            assert abs(v) < 1e-9  # H_prev=0, z0.5, h=0  H=0.5*0+0.5*0=0

    def test_gru_large_update_gate_preserves_state(self):
        """With very large positive z-gate, H_t  H_{t-1}: state is preserved."""
        from orbital.translation.steps.gru import GRUTranslator

        I, H, T = 1, 1, 1
        # W_z large positive: z  1  H_t  H_prev (but H_prev=0 at t=0)
        # With H_prev=0, H_t = (1-z)*h + z*0 = (1-z)*h
        # If z1 then H_t  0 regardless
        W_data = [0.0] * (3 * H * I)
        W_data[0] = 100.0  # W_z large  z1
        R_data = [0.0] * (3 * H * H)

        graph = self._make_gru_graph(I, H, T, W_data, R_data)
        table = ibis.memtable({"x0": [1.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"]})

        GRUTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y_h")
        assert result is not None
        backend = ibis.duckdb.connect()
        val = backend.execute(list(result.values())[0]).tolist()[0]
        assert abs(val) < 1e-6  # z1, H_prev=0  H0

    def test_gru_multi_step_output_is_group(self):
        """Two timesteps: output should still be H-length ValueVariablesGroup."""
        from orbital.translation.steps.gru import GRUTranslator

        I, H, T = 2, 3, 2
        W_data = [0.0] * (3 * H * I)
        R_data = [0.0] * (3 * H * H)

        graph = self._make_gru_graph(I, H, T, W_data, R_data)
        table = ibis.memtable(
            {"x0": [1.0], "x1": [2.0], "x2": [3.0], "x3": [4.0]}
        )
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup(
            {"x0": table["x0"], "x1": table["x1"], "x2": table["x2"], "x3": table["x3"]}
        )

        GRUTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y_h")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == H

    def test_gru_rejects_bidirectional(self):
        """Bidirectional GRU with 1-direction weights raises ValueError."""
        from orbital.translation.steps.gru import GRUTranslator

        I, H, T = 1, 1, 1
        W_data = [0.0] * (3 * H * I)
        R_data = [0.0] * (3 * H * H)
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, 3 * H, I], W_data)
        R_tensor = helper.make_tensor("R", TensorProto.FLOAT, [1, 3 * H, H], R_data)

        node = helper.make_node(
            "GRU",
            inputs=["X", "W", "R"],
            outputs=["", "Y_h"],
            hidden_size=H,
            direction="bidirectional",
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I])],
            [helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [None, H])],
            [W_tensor, R_tensor],
        )
        table = ibis.memtable({"x0": [1.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"]})

        with pytest.raises(ValueError, match="direction"):
            GRUTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()

    def test_gru_gate_order_zrh_regression(self):
        """Gate-order regression: verifies ONNX ZRH gate ordering is applied.

        Arrangement: only W[gate=2 (H), unit=0, input=0] = 1.0 and
        W[gate=0 (Z)] = 0 (so z_gate = sigmoid(0) = 0.5).
        All R = 0, no bias,  T=1, I=1, H=1,  x = [1.0].

        Expected under ZRH:
          z_gate   = sigmoid(0)   = 0.5
          r_gate   = sigmoid(0)   = 0.5
          h_tilde  = tanh(1.0)    ≈ 0.7616   (W[2]=1, r*H_prev*R = 0)
          H_out    = (1-0.5)*tanh(1.0) + 0.5*0 = 0.5*tanh(1.0) ≈ 0.3808

        A different gate ordering (e.g. ZHR) would map W[2] to the R-gate
        instead of H-gate, yielding h_tilde = tanh(0) = 0  =>  H_out = 0.
        """
        import math
        from orbital.translation.steps.gru import GRUTranslator

        I, H, T = 1, 1, 1
        W_data = [0.0] * (3 * H * I)
        # gate 0 = Z: W[0,0,0] = 0  → z = sigmoid(0) = 0.5
        # gate 1 = R: W[1,0,0] = 0  → r = sigmoid(0) = 0.5
        # gate 2 = H: W[2,0,0] = 1  → h_tilde = tanh(x)  (the discriminating weight)
        W_data[2 * H * I] = 1.0
        R_data = [0.0] * (3 * H * H)

        graph = self._make_gru_graph(I, H, T, W_data, R_data)
        table = ibis.memtable({"x0": [1.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"]})

        GRUTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y_h")
        assert isinstance(result, ValueVariablesGroup)
        backend = ibis.duckdb.connect()
        h_out = backend.execute(list(result.values())[0]).tolist()[0]

        # Expected: (1-0.5)*tanh(1.0) = 0.5*tanh(1.0)
        expected = 0.5 * math.tanh(1.0)
        assert abs(h_out - expected) < 1e-6, (
            f"GRU gate-order regression failed: got {h_out}, expected {expected}. "
            "If this fails, gate index 2 is not being used as the H-gate (ZRH ordering violated)."
        )

    def test_gru_linear_before_reset_1(self):
        """linear_before_reset=1: reset gate applied after recurrent transform.

        Arrangement: I=1, H=1, T=1, x=[1.0].
          W_z=0, W_r=0, W_h=1
          R_z=0, R_r=0, R_h=2
          B=[0]*6  (no bias), H_prev=0.

        With lbr=0 (default):
          pre_h = x*W_h + (r*H_prev)*R_h = 1 + 0 = 1    → tanh(1)
        With lbr=1:
          rec_h = H_prev*R_h + Rbh = 0
          pre_h = x*W_h + r_gate*rec_h + Wbh = 1 + 0.5*0 = 1 → tanh(1)

        These coincide when H_prev=0.  To distinguish, set R_h=2 and give
        a non-zero H_prev by running two time steps, then compare the h-gate
        computation at t=1.

        Easier closed-form test: use a bias Rbh=1.
          With lbr=0: pre_h = x*W_h + (r*H_prev)*R_h + Wbh + Rbh
                             = 1 + 0 + 0 + 1 = 2
          With lbr=1: rec_h = H_prev*R_h + Rbh = 0*2 + 1 = 1
                      pre_h = x*W_h + r*rec_h + Wbh = 1 + 0.5*1 + 0 = 1.5
                      h_tilde = tanh(1.5)
        So lbr=1 with Rbh=1 gives h_tilde=tanh(1.5) ≠ tanh(2). Gate produces
        H_out = (1-z)*h_tilde + z*H_prev = 0.5*tanh(1.5).
        """
        import math
        from orbital.translation.steps.gru import GRUTranslator

        I, H, T = 1, 1, 1
        W_data = [0.0] * (3 * H * I)
        W_data[2 * H * I] = 1.0        # W_h = 1
        R_data = [0.0] * (3 * H * H)
        R_data[2 * H * H] = 2.0        # R_h = 2

        # B layout: [Wbz, Wbr, Wbh, Rbz, Rbr, Rbh]
        # Set Rbh (index 5) = 1.0 to distinguish lbr=0 vs lbr=1
        B_data = [0.0] * (6 * H)
        B_data[5] = 1.0                # Rbh = 1

        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, 3 * H, I], W_data)
        R_tensor = helper.make_tensor("R", TensorProto.FLOAT, [1, 3 * H, H], R_data)
        B_tensor = helper.make_tensor("B", TensorProto.FLOAT, [1, 6 * H], B_data)

        node = helper.make_node(
            "GRU",
            inputs=["X", "W", "R", "B"],
            outputs=["", "Y_h"],
            hidden_size=H,
            linear_before_reset=1,
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I])],
            [helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [None, H])],
            [W_tensor, R_tensor, B_tensor],
        )
        table = ibis.memtable({"x0": [1.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"]})

        GRUTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y_h")
        assert isinstance(result, ValueVariablesGroup)
        backend = ibis.duckdb.connect()
        h_out = backend.execute(list(result.values())[0]).tolist()[0]

        # lbr=1: rec_h=H_prev*2+1=1, pre_h=1+0.5*1=1.5, h_tilde=tanh(1.5)
        # z=sigmoid(0)=0.5, H_out=(1-0.5)*tanh(1.5)+0.5*0=0.5*tanh(1.5)
        expected = 0.5 * math.tanh(1.5)
        assert abs(h_out - expected) < 1e-6, (
            f"GRU linear_before_reset=1 failed: got {h_out}, expected {expected}."
        )


# ---------------------------------------------------------------------------
# Unit tests: ExpTranslator
# ---------------------------------------------------------------------------


class TestRNNTranslator:
    """Tests for the ONNX RNN translator (forward-direction unrolling)."""


    def _make_rnn_graph(self, I, H, T, W_data, R_data, B_data=None):
        """Build a minimal ONNX RNN graph with Y_h output."""
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, H, I], W_data)
        R_tensor = helper.make_tensor("R", TensorProto.FLOAT, [1, H, H], R_data)
        inputs = ["X", "W", "R"]
        inits = [W_tensor, R_tensor]
        if B_data is not None:
            B_tensor = helper.make_tensor("B", TensorProto.FLOAT, [1, 2 * H], B_data)
            inputs.append("B")
            inits.append(B_tensor)
        node = helper.make_node(
            "RNN",
            inputs=inputs,
            outputs=["", "Y_h"],
            hidden_size=H,
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I])],
            [helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [None, H])],
            inits,
        )
        return graph

    def test_rnn_registered(self):
        from orbital.translation.steps.rnn import RNNTranslator
        assert TRANSLATORS.get("RNN") is RNNTranslator

    def test_rnn_zero_weights_zero_output(self):
        """All-zero weights and input: H_t = tanh(0) = 0."""
        from orbital.translation.steps.rnn import RNNTranslator

        I, H, T = 2, 2, 1
        W_data = [0.0] * (H * I)
        R_data = [0.0] * (H * H)

        graph = self._make_rnn_graph(I, H, T, W_data, R_data)
        table = ibis.memtable({"x0": [0.0], "x1": [0.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"], "x1": table["x1"]})

        RNNTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y_h")
        assert isinstance(result, ValueVariablesGroup)
        backend = ibis.duckdb.connect()
        for v in result.values():
            assert abs(backend.execute(v).tolist()[0]) < 1e-9

    def test_rnn_identity_weight_tanh(self):
        """W=identity, R=0, B=0, x=[1,0]: H_1 = tanh(1) at h=0, tanh(0) at h=1."""
        import math
        from orbital.translation.steps.rnn import RNNTranslator

        I, H, T = 2, 2, 1
        # W[h, i] = 1 if h==i else 0 (identity-like)
        W_data = [1.0, 0.0, 0.0, 1.0]  # H=2, I=2
        R_data = [0.0] * (H * H)

        graph = self._make_rnn_graph(I, H, T, W_data, R_data)
        table = ibis.memtable({"x0": [1.0], "x1": [0.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"], "x1": table["x1"]})

        RNNTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y_h")
        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        assert abs(vals[0] - math.tanh(1.0)) < 1e-6
        assert abs(vals[1] - math.tanh(0.0)) < 1e-9

    def test_rnn_direction_bidirectional_raises(self):
        """Bidirectional RNN with 1-direction weights raises ValueError."""
        from orbital.translation.steps.rnn import RNNTranslator

        I, H, T = 1, 1, 1
        W_data = [0.0] * (H * I)
        R_data = [0.0] * (H * H)
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, H, I], W_data)
        R_tensor = helper.make_tensor("R", TensorProto.FLOAT, [1, H, H], R_data)
        node = helper.make_node(
            "RNN",
            inputs=["X", "W", "R"],
            outputs=["", "Y_h"],
            hidden_size=H,
            direction="bidirectional",
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I])],
            [helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [None, H])],
            [W_tensor, R_tensor],
        )
        table = ibis.memtable({"x0": [1.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"]})
        t = RNNTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        with pytest.raises(ValueError, match="direction"):
            t.process()


# ---------------------------------------------------------------------------
# Unit tests: MultiHeadAttentionTranslator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit tests: MultiHeadAttentionTranslator
# ---------------------------------------------------------------------------


class TestMultiHeadAttentionTranslator:
    """Tests for the ONNX MultiHeadAttention (com.microsoft contrib) translator."""

    backend = ibis.duckdb.connect()

    def _make_mha_graph(self, T_q, D, num_heads, bias_flat=None):
        """Build a minimal ONNX MultiHeadAttention graph (self-attention)."""
        inputs = ["Q", "", "", "B"] if bias_flat is not None else ["Q", "", ""]
        node = helper.make_node(
            "MultiHeadAttention",
            inputs=inputs,
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
        )
        inits = []
        if bias_flat is not None:
            b_tensor = helper.make_tensor(
                "B", TensorProto.FLOAT, [len(bias_flat)], bias_flat
            )
            inits.append(b_tensor)
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("Q", TensorProto.FLOAT, [None, T_q * D])],
            [
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [None, T_q * D]
                )
            ],
            inits,
        )
        return graph

    def test_mha_registered(self):
        from orbital.translation.steps.multiheadattention import (
            MultiHeadAttentionTranslator,
        )
        assert TRANSLATORS.get("MultiHeadAttention") is MultiHeadAttentionTranslator

    def test_mha_self_attention_uniform_weights(self):
        """Self-attention: identity Q=K=V, zero bias, uniform scores → average of V."""
        from orbital.translation.steps.multiheadattention import (
            MultiHeadAttentionTranslator,
        )

        # Single head, 2 timesteps, D=2 → head_dim=2
        # Q = K = V = [[1, 0], [0, 1]]
        # bias = [0] * 6
        T_q, D, num_heads = 2, 2, 1
        bias_flat = [0.0] * (3 * D)

        graph = self._make_mha_graph(T_q, D, num_heads, bias_flat)
        table = ibis.memtable(
            {"q0": [1.0], "q1": [0.0], "q2": [0.0], "q3": [1.0]}
        )
        q_group = ValueVariablesGroup(
            {
                "q0": table["q0"],
                "q1": table["q1"],
                "q2": table["q2"],
                "q3": table["q3"],
            }
        )

        variables = GraphVariables(ibis.memtable({"Q": [0.0]}), graph)
        variables["Q"] = q_group

        MultiHeadAttentionTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        # Output has T_q * D = 4 entries
        assert len(result) == T_q * D

    def test_mha_no_bias_no_hidden_size_raises(self):
        """Without bias and without hidden_size attribute, must raise NotImplementedError."""
        from orbital.translation.steps.multiheadattention import (
            MultiHeadAttentionTranslator,
        )

        T_q, D, num_heads = 2, 2, 1
        # No bias
        node = helper.make_node(
            "MultiHeadAttention",
            inputs=["Q"],
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("Q", TensorProto.FLOAT, [None, T_q * D])],
            [
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [None, T_q * D]
                )
            ],
            [],
        )
        table = ibis.memtable({"q0": [1.0], "q1": [0.0], "q2": [0.0], "q3": [1.0]})
        q_group = ValueVariablesGroup(
            {"q0": table["q0"], "q1": table["q1"], "q2": table["q2"], "q3": table["q3"]}
        )
        variables = GraphVariables(ibis.memtable({"Q": [0.0]}), graph)
        variables["Q"] = q_group
        t = MultiHeadAttentionTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        with pytest.raises(NotImplementedError, match="hidden size"):
            t.process()

    def test_mha_hidden_size_attribute(self):
        """When bias is absent but hidden_size attribute is set, translation succeeds."""
        from orbital.translation.steps.multiheadattention import (
            MultiHeadAttentionTranslator,
        )

        T_q, D, num_heads = 2, 4, 2  # head_dim = 2
        node = helper.make_node(
            "MultiHeadAttention",
            inputs=["Q"],
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
            hidden_size=D,
        )
        graph = _make_graph_with_inits(
            node,
            [
                helper.make_tensor_value_info(
                    "Q", TensorProto.FLOAT, [None, T_q * D]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [None, T_q * D]
                )
            ],
            [],
        )
        cols = {f"q{i}": [float(i % 2)] for i in range(T_q * D)}
        table = ibis.memtable(cols)
        q_group = ValueVariablesGroup({k: table[k] for k in cols})
        variables = GraphVariables(ibis.memtable({"Q": [0.0]}), graph)
        variables["Q"] = q_group

        MultiHeadAttentionTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == T_q * D

    def test_multiheadattention_cross_attention_different_value_dim(self):
        """After the BUG-4 fix, D_v is inferred from V columns, not set equal to D."""
        from orbital.translation.steps.multiheadattention import (
            MultiHeadAttentionTranslator,
        )

        # Q: T_q=2 timesteps, D=8 features
        # K: T_k=3 timesteps, D=8 features
        # V: T_k=3 timesteps, D_v=4 features (D_v != D)
        # D supplied via hidden_size attribute so D_v inference from V shape is exercised.
        T_q, T_k, D, D_v, num_heads = 2, 3, 8, 4, 2
        node = helper.make_node(
            "MultiHeadAttention",
            inputs=["Q", "K", "V"],
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
            hidden_size=D,
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("Q", TensorProto.FLOAT, [None, T_q * D])],
            [
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [None, T_q * D_v]
                )
            ],
            [],
        )

        q_cols = {f"q{i}": [float(i % 4) / 4.0] for i in range(T_q * D)}
        k_cols = {f"k{i}": [float(i % 4) / 4.0] for i in range(T_k * D)}
        v_cols = {f"v{i}": [float(i % 2) / 2.0] for i in range(T_k * D_v)}
        table = ibis.memtable({**q_cols, **k_cols, **v_cols})

        variables = GraphVariables(ibis.memtable({"Q": [0.0]}), graph)
        variables["Q"] = ValueVariablesGroup({k: table[k] for k in q_cols})
        variables["K"] = ValueVariablesGroup({k: table[k] for k in k_cols})
        variables["V"] = ValueVariablesGroup({k: table[k] for k in v_cols})

        MultiHeadAttentionTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        # BUG-4 fix: output has T_q * D_v = 2 * 4 = 8 columns, not T_q * D = 2 * 8 = 16
        assert len(result) == T_q * D_v


# ---------------------------------------------------------------------------
# Unit tests: AttentionTranslator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit tests: AttentionTranslator
# ---------------------------------------------------------------------------


class TestAttentionTranslator:
    """Tests for the ONNX Attention (com.microsoft fused-QKV) translator."""

    backend = ibis.duckdb.connect()

    def _make_attention_graph(self, T, I_size, H, num_heads, w_flat, bias_flat=None):
        """Build a minimal ONNX Attention graph."""
        inputs = ["X", "W", "B"] if bias_flat is not None else ["X", "W"]
        node = helper.make_node(
            "Attention",
            inputs=inputs,
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
        )
        inits = [
            helper.make_tensor("W", TensorProto.FLOAT, [I_size, 3 * H], w_flat),
        ]
        if bias_flat is not None:
            inits.append(
                helper.make_tensor("B", TensorProto.FLOAT, [3 * H], bias_flat)
            )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I_size])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, T * H])],
            inits,
        )
        return graph

    def test_attention_registered(self):
        from orbital.translation.steps.attention import AttentionTranslator
        assert TRANSLATORS.get("Attention") is AttentionTranslator

    def test_attention_identity_weight_single_timestep(self):
        """Single timestep: identity QKV weights, no bias → softmax of one score = 1.0."""
        from orbital.translation.steps.attention import AttentionTranslator

        # T=1, I=2, H=2, num_heads=1, head_dim=2
        # W = identity-like: just a 2x6 matrix (first two cols = Q, next = K, last = V)
        # Use all-zero weights for simplicity → output should be all zeros
        T, I_size, H, num_heads = 1, 2, 2, 1
        w_flat = [0.0] * (I_size * 3 * H)

        graph = self._make_attention_graph(T, I_size, H, num_heads, w_flat)
        cols = {"x0": [1.0], "x1": [2.0]}
        table = ibis.memtable(cols)
        x_group = ValueVariablesGroup({k: table[k] for k in cols})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = x_group

        AttentionTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        # T * H = 2 output columns
        assert len(result) == T * H
        # all weights zero → all projections zero → output zero
        for expr in result.values():
            val = self.backend.execute(expr).item()
            assert abs(val) < 1e-9

    def test_attention_multi_head_output_shape(self):
        """Multi-head: output group has T * H entries."""
        from orbital.translation.steps.attention import AttentionTranslator

        T, I_size, H, num_heads = 2, 4, 4, 2  # head_dim = 2
        import random
        random.seed(42)
        w_flat = [random.gauss(0, 0.1) for _ in range(I_size * 3 * H)]

        graph = self._make_attention_graph(T, I_size, H, num_heads, w_flat)
        cols = {f"x{i}": [float(i) / 10] for i in range(T * I_size)}
        table = ibis.memtable(cols)
        x_group = ValueVariablesGroup({k: table[k] for k in cols})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = x_group

        AttentionTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == T * H

    def test_attention_mask_index_raises(self):
        """Passing mask_index (input[3]) must raise NotImplementedError."""
        from orbital.translation.steps.attention import AttentionTranslator

        T, I_size, H, num_heads = 1, 2, 2, 1
        w_flat = [0.0] * (I_size * 3 * H)
        mask_tensor = helper.make_tensor("M", TensorProto.INT32, [1], [1])
        node = helper.make_node(
            "Attention",
            inputs=["X", "W", "", "M"],
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
        )
        inits = [
            helper.make_tensor("W", TensorProto.FLOAT, [I_size, 3 * H], w_flat),
            mask_tensor,
        ]
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I_size])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, T * H])],
            inits,
        )
        cols = {"x0": [1.0], "x1": [2.0]}
        table = ibis.memtable(cols)
        x_group = ValueVariablesGroup({k: table[k] for k in cols})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = x_group
        with pytest.raises(NotImplementedError, match="mask_index"):
            AttentionTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()

    def test_attention_past_input_raises_not_implemented(self):
        """Attention: providing a past KV-state input (input[4]) must raise NotImplementedError."""
        from orbital.translation.steps.attention import AttentionTranslator

        T, I_size, H, num_heads = 1, 2, 2, 1
        w_flat = [0.0] * (I_size * 3 * H)
        # input[2]=bias absent, input[3]=mask_index absent, input[4]=past present
        past_tensor = helper.make_tensor("past", TensorProto.FLOAT, [1], [0.0])
        node = helper.make_node(
            "Attention",
            inputs=["X", "W", "", "", "past"],
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
        )
        inits = [
            helper.make_tensor("W", TensorProto.FLOAT, [I_size, 3 * H], w_flat),
            past_tensor,
        ]
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I_size])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, T * H])],
            inits,
        )
        cols = {"x0": [1.0], "x1": [2.0]}
        table = ibis.memtable(cols)
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({k: table[k] for k in cols})

        with pytest.raises(NotImplementedError, match=r"past \(input\[4\]\)"):
            AttentionTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()


# ---------------------------------------------------------------------------
# Unit tests: MeanVarianceNormalizationTranslator
# ---------------------------------------------------------------------------


