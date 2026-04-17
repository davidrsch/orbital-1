"""Tests for Transpose/Flatten/Squeeze/Unsqueeze/Shape pipeline step translators."""

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


from conftest import make_graph_with_inits as _make_graph_with_inits


class TestTransposeTranslator:
    """Tests for TransposeTranslator."""

    def test_transpose_registered(self):
        """Verify TransposeTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.transpose import TransposeTranslator

        assert TRANSLATORS.get("Transpose") is TransposeTranslator

    def test_transpose_passthrough(self):
        """Transpose is a pass-through for column groups."""
        table = ibis.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Transpose <perm: ints = [1, 0]> (input)
            }
        """)
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.transpose import TransposeTranslator

        t = TransposeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert set(result.keys()) == {"a", "b"}


class TestFlattenTranslator:
    """Tests for FlattenTranslator."""

    def test_flatten_registered(self):
        """Verify FlattenTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.flatten import FlattenTranslator

        assert TRANSLATORS.get("Flatten") is FlattenTranslator

    def test_flatten_axis1_passthrough(self):
        """Flatten at axis=1 is a pass-through (the only supported case)."""
        table = ibis.memtable({"a": [1.0], "b": [2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Flatten <axis: int = 1> (input)
            }
        """)
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.flatten import FlattenTranslator

        t = FlattenTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert set(result.keys()) == {"a", "b"}

    def test_flatten_non_axis1_raises(self):
        """Flatten at axis != 1 raises NotImplementedError."""
        table = ibis.memtable({"x": [1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Flatten <axis: int = 0> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.flatten import FlattenTranslator

        t = FlattenTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        with pytest.raises(NotImplementedError, match="axis=1"):
            t.process()


class TestSqueezeTranslator:
    """Tests for SqueezeTranslator."""

    def test_squeeze_registered(self):
        from orbital.translation.steps.squeeze import SqueezeTranslator

        assert TRANSLATORS.get("Squeeze") is SqueezeTranslator

    def test_squeeze_passthrough(self):
        """Squeeze is a pass-through in columnar representation."""
        table = ibis.memtable({"x": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Squeeze(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.squeeze import SqueezeTranslator

        t = SqueezeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [
            1.0,
            2.0,
            3.0,
        ]


class TestUnsqueezeTranslator:
    """Tests for UnsqueezeTranslator."""

    def test_unsqueeze_registered(self):
        from orbital.translation.steps.squeeze import UnsqueezeTranslator

        assert TRANSLATORS.get("Unsqueeze") is UnsqueezeTranslator

    def test_unsqueeze_passthrough(self):
        """Unsqueeze is a pass-through in columnar representation."""
        table = ibis.memtable({"x": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <int64[1] axes = {0}>
            {
                output = Unsqueeze(x, axes)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.squeeze import UnsqueezeTranslator

        t = UnsqueezeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [
            1.0,
            2.0,
            3.0,
        ]


class TestShapeTranslator:
    """Tests for ShapeTranslator."""

    def test_shape_registered(self):
        from orbital.translation.steps.shape import ShapeTranslator

        assert TRANSLATORS.get("Shape") is ShapeTranslator

    def test_shape_raises(self):
        """Shape with start=0 (default) still raises NotImplementedError (batch dim)."""
        node = helper.make_node("Shape", inputs=["x"], outputs=["y"])
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.INT64, [None])],
            [],
        )
        table = ibis.memtable({"x": [1.0]})
        variables = GraphVariables(table, graph)
        variables["x"] = table["x"]
        from orbital.translation.steps.shape import ShapeTranslator

        t = ShapeTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        with pytest.raises(NotImplementedError):
            t.process()

    def test_start_1_returns_feature_count(self):
        """Shape with start=1 returns an ibis literal equal to the feature-column count."""
        from orbital.translation.steps.shape import ShapeTranslator

        # Input group has 3 feature columns; Shape(start=1) should return 3
        data_tbl = ibis.memtable({"a": [1.0], "b": [2.0], "c": [3.0]})
        node = helper.make_node("Shape", inputs=["x"], outputs=["y"], start=1)
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.INT64, [None])],
            [],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = ValueVariablesGroup(
            {"a": data_tbl["a"], "b": data_tbl["b"], "c": data_tbl["c"]}
        )
        ShapeTranslator(
            data_tbl, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("y")
        backend = ibis.duckdb.connect()
        assert int(backend.execute(result)) == 3

    def test_start_0_raises(self):
        """Shape with start=0 (includes batch dimension) raises NotImplementedError."""
        from orbital.translation.steps.shape import ShapeTranslator

        node = helper.make_node("Shape", inputs=["x"], outputs=["y"], start=0)
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.INT64, [None])],
            [],
        )
        table = ibis.memtable({"x": [1.0]})
        variables = GraphVariables(table, graph)
        variables["x"] = table["x"]
        with pytest.raises(NotImplementedError):
            ShapeTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()


# ---------------------------------------------------------------------------
# Unit tests: TileTranslator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit tests: TileTranslator
# ---------------------------------------------------------------------------
