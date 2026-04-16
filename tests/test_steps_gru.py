"""Tests for the GRU pipeline step translator."""

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
