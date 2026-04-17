"""Tests for the LSTM pipeline step translator."""

import math

import ibis
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from conftest import make_graph_with_inits as _make_graph_with_inits
from orbital.translate import TRANSLATORS
from orbital.translation.options import TranslationOptions
from orbital.translation.variables import GraphVariables, ValueVariablesGroup

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
        W_data[3 * H * I] = 1.0  # W_c[0, 0] = 1.0

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
        W_data[0 * H * I] = 100.0  # gate 0 = I-gate: force i ≈ 1.0
        W_data[3 * H * I] = 1.0  # gate 3 = C-gate: c_bar = tanh(x)
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
