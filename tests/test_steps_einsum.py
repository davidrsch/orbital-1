"""Tests for the Einsum pipeline step translator."""

import ibis
import numpy as np
import pytest
from onnx import TensorProto, helper

from conftest import make_graph_with_inits as _make_graph_with_inits
from orbital.translate import TRANSLATORS
from orbital.translation.options import TranslationOptions
from orbital.translation.variables import GraphVariables, ValueVariablesGroup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_einsum_graph(equation, x_shape, w_data, w_shape, out_shape):
    """Build a minimal ONNX graph with a single Einsum node."""
    w_tensor = helper.make_tensor("W", TensorProto.FLOAT, w_shape, w_data)
    node = helper.make_node("Einsum", inputs=["X", "W"], outputs=["Y"], equation=equation)
    x_flat_len = 1
    for d in x_shape:
        if d is not None:
            x_flat_len *= d
    return _make_graph_with_inits(
        node,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, x_flat_len])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape)],
        [w_tensor],
    )


# ---------------------------------------------------------------------------
# Unit tests: EinsumTranslator
# ---------------------------------------------------------------------------


class TestEinsumTranslator:
    """Tests for the ONNX Einsum translator."""

    def test_einsum_registered(self):
        from orbital.translation.steps.einsum import EinsumTranslator
        assert TRANSLATORS.get("Einsum") is EinsumTranslator

    def test_einsum_matmul_ab_bc_ac(self):
        """'ab,bc->ac': 1×2 input multiplied by 2×3 weight = 1×3 output."""
        from orbital.translation.steps.einsum import EinsumTranslator

        # X = [[1, 2]],  W = [[1,0,0],[0,1,0]]  (2×3 identity-ish)
        # Expected: [1, 2, 0] ... no, W:
        #   W = [[1, 2, 3], [4, 5, 6]]
        # X @ W = [1*1+2*4, 1*2+2*5, 1*3+2*6] = [9, 12, 15]
        w_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # shape (2, 3), row-major
        graph = _make_einsum_graph("ab,bc->ac", [1, 2], w_data, [2, 3], [None, 3])

        table = ibis.memtable({"x0": [1.0], "x1": [2.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"], "x1": table["x1"]})

        EinsumTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 3

        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        assert vals == pytest.approx([9.0, 12.0, 15.0])

    def test_einsum_single_output(self):
        """'ab,bc->ac' with output_dim=1 returns a scalar column, not a group."""
        from orbital.translation.steps.einsum import EinsumTranslator

        # X = [[3, 4]], W = [[2], [1]]  → Y = [3*2+4*1] = [10]
        w_data = [2.0, 1.0]  # shape (2, 1)
        graph = _make_einsum_graph("ab,bc->ac", [1, 2], w_data, [2, 1], [None, 1])

        table = ibis.memtable({"x0": [3.0], "x1": [4.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"], "x1": table["x1"]})

        EinsumTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y")
        assert not isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        val = backend.execute(result).tolist()[0]
        assert val == pytest.approx(10.0)

    def test_einsum_identity_weight(self):
        """Identity weight matrix: output equals input."""
        from orbital.translation.steps.einsum import EinsumTranslator

        # W = I_3  → Y = X
        w_data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # shape (3, 3)
        graph = _make_einsum_graph("ab,bc->ac", [1, 3], w_data, [3, 3], [None, 3])

        table = ibis.memtable({"x0": [5.0], "x1": [-3.0], "x2": [7.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup(
            {"x0": table["x0"], "x1": table["x1"], "x2": table["x2"]}
        )

        EinsumTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        assert vals == pytest.approx([5.0, -3.0, 7.0])

    def test_einsum_rejects_dynamic_second_input(self):
        """Dynamic (non-initializer) second input raises NotImplementedError."""
        from orbital.translation.steps.einsum import EinsumTranslator

        w_data = [1.0, 0.0, 0.0, 1.0]  # shape (2, 2)
        graph = _make_einsum_graph("ab,bc->ac", [1, 2], w_data, [2, 2], [None, 2])

        table = ibis.memtable({"x0": [1.0], "x1": [2.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"], "x1": table["x1"]})
        # Simulate a dynamic W by planting it in variables (not as an initializer)
        variables["W"] = ValueVariablesGroup({"w0": table["x0"], "w1": table["x1"]})

        with pytest.raises(NotImplementedError, match="constant weight"):
            EinsumTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()

    def test_einsum_rejects_unsupported_equation(self):
        """Unsupported equation raises NotImplementedError."""
        from orbital.translation.steps.einsum import EinsumTranslator

        w_data = [1.0, 0.0, 0.0, 1.0]
        graph = _make_einsum_graph("ij,jk->ik", [1, 2], w_data, [2, 2], [None, 2])
        # Patch the equation attribute to something unsupported
        graph.node[0].attribute[0].s = b"ijk,klm->ijlm"

        table = ibis.memtable({"x0": [1.0], "x1": [2.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"], "x1": table["x1"]})

        with pytest.raises(NotImplementedError):
            EinsumTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()

    def test_einsum_batch_style_ellipsis_equation(self):
        """'...b,bc->...c' (normalised to same matmul): same result as 'ab,bc->ac'."""
        from orbital.translation.steps.einsum import EinsumTranslator

        # X = [[1, 2]],  W = [[1, 2, 3], [4, 5, 6]]  → [9, 12, 15]
        w_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        graph = _make_einsum_graph("...b,bc->...c", [1, 2], w_data, [2, 3], [None, 3])

        table = ibis.memtable({"x0": [1.0], "x1": [2.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"], "x1": table["x1"]})

        EinsumTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        assert vals == pytest.approx([9.0, 12.0, 15.0])
