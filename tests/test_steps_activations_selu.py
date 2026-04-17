"""Tests for Swish ThresholdedRelu Shrink pipeline step translators."""

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


class TestSwishTranslator:
    """Tests for the ONNX Swish translator (x * sigmoid(x))."""

    def test_swish_registered(self):
        from orbital.translation.steps.swish import SwishTranslator

        assert TRANSLATORS.get("Swish") is SwishTranslator

    def test_swish_single_column(self):
        """Swish(x) = x / (1 + exp(-x))."""
        table = ibis.memtable({"x": [0.0, 1.0, -2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Swish(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.swish import SwishTranslator

        SwishTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        import math

        def swish(x):
            return x / (1.0 + math.exp(-x))

        expected = [swish(v) for v in [0.0, 1.0, -2.0]]
        for got, exp in zip(result, expected):
            assert abs(got - exp) < 1e-9

    def test_swish_zero_is_zero(self):
        """Swish(0) = 0 * sigmoid(0) = 0."""
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Swish(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.swish import SwishTranslator

        SwishTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [0.0]


# ---------------------------------------------------------------------------
# Unit tests: ThresholdedReluTranslator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit tests: ThresholdedReluTranslator
# ---------------------------------------------------------------------------


class TestThresholdedReluTranslator:
    """Tests for the ONNX ThresholdedRelu translator (x if x > alpha else 0)."""

    def test_thresholdedrelu_registered(self):
        from orbital.translation.steps.thresholdedrelu import ThresholdedReluTranslator

        assert TRANSLATORS.get("ThresholdedRelu") is ThresholdedReluTranslator

    def test_thresholdedrelu_default_alpha(self):
        """ThresholdedRelu with default alpha=1.0: 0 for x<=1, x for x>1."""
        table = ibis.memtable({"x": [0.5, 1.0, 1.5, 2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ThresholdedRelu(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.thresholdedrelu import ThresholdedReluTranslator

        ThresholdedReluTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [0.0, 0.0, 1.5, 2.0]

    def test_thresholdedrelu_custom_alpha(self):
        """ThresholdedRelu with alpha=2.0."""
        table = ibis.memtable({"x": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ThresholdedRelu <alpha: float = 2.0> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.thresholdedrelu import ThresholdedReluTranslator

        ThresholdedReluTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [0.0, 0.0, 3.0]

    def test_thresholdedrelu_group(self):
        """ThresholdedRelu applied element-wise to a column group."""
        table = ibis.memtable({"a": [0.5, 2.0], "b": [1.5, -1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ThresholdedRelu(x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.thresholdedrelu import ThresholdedReluTranslator

        ThresholdedReluTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        a_vals = list(backend.execute(result["a"]))
        b_vals = list(backend.execute(result["b"]))
        assert a_vals == [0.0, 2.0]
        assert b_vals == [1.5, 0.0]


# ---------------------------------------------------------------------------
# Unit tests: RMSNormalizationTranslator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit tests: ShrinkTranslator
# ---------------------------------------------------------------------------


class TestShrinkTranslator:
    """Tests for ShrinkTranslator."""

    def test_shrink_registered(self):
        from orbital.translation.steps.shrink import ShrinkTranslator

        assert TRANSLATORS.get("Shrink") is ShrinkTranslator

    def test_shrink_below_neg_lambd(self):
        """x=-2, lambd=0.5, bias=1.0: y = -2 + 1 = -1."""
        table = ibis.memtable({"x": [-2.0]})
        scale_node = helper.make_node(
            "Shrink", inputs=["x"], outputs=["y"], lambd=0.5, bias=1.0
        )
        graph = _make_graph_with_inits(
            scale_node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            [],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = table["x"]
        from orbital.translation.steps.shrink import ShrinkTranslator

        ShrinkTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("y")
        backend = ibis.duckdb.connect()
        assert abs(list(backend.execute(result))[0] - (-1.0)) < 1e-6

    def test_shrink_in_dead_zone(self):
        """x=0.3, lambd=0.5: y = 0 (inside dead zone)."""
        table = ibis.memtable({"x": [0.3]})
        scale_node = helper.make_node("Shrink", inputs=["x"], outputs=["y"], lambd=0.5)
        graph = _make_graph_with_inits(
            scale_node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            [],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = table["x"]
        from orbital.translation.steps.shrink import ShrinkTranslator

        ShrinkTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("y")
        backend = ibis.duckdb.connect()
        assert abs(list(backend.execute(result))[0]) < 1e-6

    def test_shrink_above_lambd(self):
        """x=2, lambd=0.5, bias=0.5: y = x - bias = 1.5."""
        table = ibis.memtable({"x": [2.0]})
        scale_node = helper.make_node(
            "Shrink", inputs=["x"], outputs=["y"], lambd=0.5, bias=0.5
        )
        graph = _make_graph_with_inits(
            scale_node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            [],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = table["x"]
        from orbital.translation.steps.shrink import ShrinkTranslator

        ShrinkTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("y")
        backend = ibis.duckdb.connect()
        assert abs(list(backend.execute(result))[0] - 1.5) < 1e-6


# ---------------------------------------------------------------------------
# Unit tests: ModTranslator
# ---------------------------------------------------------------------------
