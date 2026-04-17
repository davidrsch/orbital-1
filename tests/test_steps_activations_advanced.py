"""Tests for activations pipeline step translators."""

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


class TestEluTranslator:
    """Tests for EluTranslator."""

    def test_elu_registered(self):
        from orbital.translation.steps.elu import EluTranslator

        assert TRANSLATORS.get("Elu") is EluTranslator

    def test_elu_positive_unchanged(self):
        """ELU leaves positive values unchanged."""
        table = ibis.memtable({"x": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Elu(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.elu import EluTranslator

        t = EluTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [1.0, 2.0, 3.0]

    def test_elu_negative_with_default_alpha(self):
        """ELU with negative input uses alpha*(exp(x)-1) with alpha=1.0."""
        import math

        table = ibis.memtable({"x": [-1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Elu(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.elu import EluTranslator

        t = EluTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        expected = 1.0 * (math.exp(-1.0) - 1.0)
        assert abs(val - expected) < 1e-9


class TestSeluTranslator:
    """Tests for SeluTranslator."""

    def test_selu_registered(self):
        from orbital.translation.steps.selu import SeluTranslator

        assert TRANSLATORS.get("Selu") is SeluTranslator

    def test_selu_positive(self):
        """SELU of positive input = gamma * x."""
        import math

        gamma = 1.0507009873554805
        table = ibis.memtable({"x": [1.0, 2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Selu(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.selu import SeluTranslator

        t = SeluTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert abs(result[0] - gamma * 1.0) < 1e-9
        assert abs(result[1] - gamma * 2.0) < 1e-9

    def test_selu_negative(self):
        """SELU of negative input = gamma*(alpha*(exp(x)-1))."""
        import math

        alpha = 1.6732631921768188
        gamma = 1.0507009873554805
        table = ibis.memtable({"x": [-1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Selu(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.selu import SeluTranslator

        t = SeluTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = backend.execute(variables.peek_variable("output"))[0]
        expected = gamma * alpha * (math.exp(-1.0) - 1.0)
        assert abs(result - expected) < 1e-9


class TestCeluTranslator:
    """Tests for CeluTranslator."""

    def test_celu_registered(self):
        from orbital.translation.steps.celu import CeluTranslator

        assert TRANSLATORS.get("Celu") is CeluTranslator

    def test_celu_positive_unchanged(self):
        """CELU leaves positive values unchanged (max(0,x) + min(0,neg) = x + 0)."""
        table = ibis.memtable({"x": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Celu(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.celu import CeluTranslator

        t = CeluTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [
            1.0,
            2.0,
            3.0,
        ]

    def test_celu_zero_input(self):
        """CELU of 0 = 0."""
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Celu(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.celu import CeluTranslator

        t = CeluTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        assert abs(backend.execute(variables.peek_variable("output"))[0]) < 1e-9


class TestHardSwishTranslator:
    """Tests for HardSwishTranslator."""

    def test_hardswish_registered(self):
        from orbital.translation.steps.hardswish import HardSwishTranslator

        assert TRANSLATORS.get("HardSwish") is HardSwishTranslator

    def test_hardswish_zero_input(self):
        """hard_swish(0) = 0 * max(0, min(1, 3/6)) = 0 * 0.5 = 0."""
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = HardSwish(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.hardswish import HardSwishTranslator

        t = HardSwishTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        assert backend.execute(variables.peek_variable("output"))[0] == 0.0

    def test_hardswish_large_positive(self):
        """For large x, hard_swish(x) ≈ x (gate = 1)."""
        table = ibis.memtable({"x": [100.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = HardSwish(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.hardswish import HardSwishTranslator

        t = HardSwishTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        assert backend.execute(variables.peek_variable("output"))[0] == 100.0


class TestHardTanhTranslator:
    """Tests for HardTanhTranslator."""

    def test_hardtanh_registered(self):
        from orbital.translation.steps.hardtanh import HardTanhTranslator

        assert TRANSLATORS.get("HardTanh") is HardTanhTranslator

    def test_hardtanh_clips_low(self):
        """HardTanh clips values below -1 to -1."""
        table = ibis.memtable({"x": [-5.0, -1.5]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = HardTanh(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.hardtanh import HardTanhTranslator

        t = HardTanhTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [-1.0, -1.0]

    def test_hardtanh_clips_high(self):
        """HardTanh clips values above 1 to 1."""
        table = ibis.memtable({"x": [1.5, 5.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = HardTanh(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.hardtanh import HardTanhTranslator

        t = HardTanhTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [1.0, 1.0]

    def test_hardtanh_passthrough_in_range(self):
        """HardTanh keeps values in [-1, 1] unchanged."""
        table = ibis.memtable({"x": [-0.5, 0.0, 0.5]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = HardTanh(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.hardtanh import HardTanhTranslator

        t = HardTanhTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [-0.5, 0.0, 0.5]

    def test_custom_bounds(self):
        """HardTanh with explicit min=-2.0 max=2.5 attributes clips at those bounds."""
        from orbital.translation.steps.hardtanh import HardTanhTranslator

        table = ibis.memtable({"x": [-3.0, 0.0, 3.0]})
        node = helper.make_node(
            "HardTanh", inputs=["x"], outputs=["output"], min=-2.0, max=2.5
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None])],
            [],
        )
        variables = GraphVariables(table, graph)
        t = HardTanhTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [-2.0, 0.0, 2.5]

    def test_default_bounds_unchanged(self):
        """HardTanh with no min/max attributes defaults to clip[-1, 1]."""
        from orbital.translation.steps.hardtanh import HardTanhTranslator

        table = ibis.memtable({"x": [-2.0, 0.0, 2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = HardTanh(x)
            }
        """)
        variables = GraphVariables(table, model)
        t = HardTanhTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [-1.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# New translator tests (Python Issues #30#35)
# ---------------------------------------------------------------------------
