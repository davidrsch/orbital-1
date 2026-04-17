"""Tests for the Trilu pipeline step translator."""

import ibis
import pytest
from onnx import TensorProto, helper

from conftest import make_graph_with_inits as _make_graph_with_inits
from orbital.translate import TRANSLATORS
from orbital.translation.options import TranslationOptions
from orbital.translation.variables import GraphVariables, ValueVariablesGroup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trilu_graph(x_data, x_shape, upper=1, k=None):
    """Build a minimal ONNX graph for Trilu with a constant initializer input."""
    x_tensor = helper.make_tensor("X", TensorProto.FLOAT, x_shape, x_data)
    inputs = ["X"]
    inits = [x_tensor]
    attrs = {"upper": upper}

    if k is not None:
        k_tensor = helper.make_tensor("K", TensorProto.INT64, [], [k])
        inputs.append("K")
        inits.append(k_tensor)

    n_out = x_shape[0] * x_shape[1]
    node = helper.make_node("Trilu", inputs=inputs, outputs=["Y"], **attrs)
    return _make_graph_with_inits(
        node,
        [],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, n_out])],
        inits,
    )


# ---------------------------------------------------------------------------
# Unit tests: TriluTranslator
# ---------------------------------------------------------------------------


class TestTriluTranslator:
    """Tests for the ONNX Trilu translator."""

    def test_trilu_registered(self):
        from orbital.translation.steps.trilu import TriluTranslator

        assert TRANSLATORS.get("Trilu") is TriluTranslator

    def test_trilu_upper_default(self):
        """upper=1 (default): elements below the main diagonal are zeroed."""
        from orbital.translation.steps.trilu import TriluTranslator

        # 3×3 matrix of ones → upper triangle (including diagonal) = 1, rest = 0
        x_data = [1.0] * 9
        graph = _make_trilu_graph(x_data, [3, 3], upper=1)

        table = ibis.memtable({"dummy": [0]})
        variables = GraphVariables(ibis.memtable({"dummy": [0]}), graph)

        TriluTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 9

        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        # Expected upper-triangular mask (row-major):
        # row0: [1,1,1], row1: [0,1,1], row2: [0,0,1]
        expected = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
        assert vals == pytest.approx(expected)

    def test_trilu_lower(self):
        """upper=0: elements above the main diagonal are zeroed."""
        from orbital.translation.steps.trilu import TriluTranslator

        x_data = [1.0] * 9
        graph = _make_trilu_graph(x_data, [3, 3], upper=0)

        table = ibis.memtable({"dummy": [0]})
        variables = GraphVariables(ibis.memtable({"dummy": [0]}), graph)

        TriluTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y")
        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        # Expected lower-triangular mask:
        # row0: [1,0,0], row1: [1,1,0], row2: [1,1,1]
        expected = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        assert vals == pytest.approx(expected)

    def test_trilu_upper_with_positive_k(self):
        """upper=1, k=1: zero out the main diagonal too."""
        from orbital.translation.steps.trilu import TriluTranslator

        x_data = [1.0] * 9
        graph = _make_trilu_graph(x_data, [3, 3], upper=1, k=1)

        table = ibis.memtable({"dummy": [0]})
        variables = GraphVariables(ibis.memtable({"dummy": [0]}), graph)

        TriluTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y")
        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        # keep col >= row+1 → strict upper triangle
        # row0: [0,1,1], row1: [0,0,1], row2: [0,0,0]
        expected = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        assert vals == pytest.approx(expected)

    def test_trilu_lower_with_negative_k(self):
        """upper=0, k=-1: keep only elements strictly below the diagonal."""
        from orbital.translation.steps.trilu import TriluTranslator

        x_data = [1.0] * 9
        graph = _make_trilu_graph(x_data, [3, 3], upper=0, k=-1)

        table = ibis.memtable({"dummy": [0]})
        variables = GraphVariables(ibis.memtable({"dummy": [0]}), graph)

        TriluTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y")
        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        # keep col <= row-1 → strict lower triangle
        # row0: [0,0,0], row1: [1,0,0], row2: [1,1,0]
        expected = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
        assert vals == pytest.approx(expected)

    def test_trilu_preserves_values(self):
        """Values inside the triangle are preserved (not just masked to 1)."""
        from orbital.translation.steps.trilu import TriluTranslator

        x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        graph = _make_trilu_graph(x_data, [3, 3], upper=1)

        table = ibis.memtable({"dummy": [0]})
        variables = GraphVariables(ibis.memtable({"dummy": [0]}), graph)

        TriluTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y")
        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        # Upper triangle keeps original values:
        # row0: [1,2,3], row1: [0,5,6], row2: [0,0,9]
        expected = [1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]
        assert vals == pytest.approx(expected)
