"""Tests for smooth activation pipeline step translators."""

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


class TestTanhTranslator:
    """Tests for TanhTranslator — see test_mlp.py for comprehensive tests."""

    def test_tanh_registered(self):
        """Verify TanhTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.tanh import TanhTranslator

        assert TRANSLATORS.get("Tanh") is TanhTranslator

    def test_tanh_correctness(self):
        """tanh(0) = 0; tanh preserves sign."""
        import math
        from orbital.translation.steps.tanh import TanhTranslator

        table = ibis.memtable({"input": [0.0, 1.0, -1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Tanh(input)
            }
        """)
        variables = GraphVariables(table, model)
        TanhTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = (
            ibis.duckdb.connect().execute(variables.peek_variable("output")).tolist()
        )
        assert result[0] == 0.0
        assert abs(result[1] - math.tanh(1.0)) < 1e-9
        assert abs(result[2] - math.tanh(-1.0)) < 1e-9

    def test_tanh_non_numeric_raises(self):
        """Tanh raises ValueError when the input variable is non-numeric."""
        import pytest
        from orbital.translation.steps.tanh import TanhTranslator

        table = ibis.memtable({"input": [1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Tanh(input)
            }
        """)
        variables = GraphVariables(table, model)
        str_table = ibis.memtable({"val": ["a", "b"]})
        variables["input"] = str_table["val"]
        with pytest.raises(ValueError, match="numeric"):
            TanhTranslator(
                table, model.node[0], variables, self.optimizer, TranslationOptions()
            ).process()


class TestSigmoidTranslator:
    """Tests for SigmoidTranslator — see test_mlp.py for comprehensive tests."""

    def test_sigmoid_registered(self):
        """Verify SigmoidTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.sigmoid import SigmoidTranslator

        assert TRANSLATORS.get("Sigmoid") is SigmoidTranslator

    def test_sigmoid_correctness(self):
        """sigmoid(0) = 0.5; sigmoid is monotonically increasing."""
        from orbital.translation.steps.sigmoid import SigmoidTranslator

        table = ibis.memtable({"input": [0.0, 1.0, -1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Sigmoid(input)
            }
        """)
        variables = GraphVariables(table, model)
        SigmoidTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = (
            ibis.duckdb.connect().execute(variables.peek_variable("output")).tolist()
        )
        assert abs(result[0] - 0.5) < 1e-9
        assert result[1] > 0.5  # sigmoid(positive) > 0.5
        assert result[2] < 0.5  # sigmoid(negative) < 0.5

    def test_sigmoid_non_numeric_raises(self):
        """Sigmoid raises ValueError when the input variable is non-numeric."""
        import pytest
        from orbital.translation.steps.sigmoid import SigmoidTranslator

        table = ibis.memtable({"input": [1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Sigmoid(input)
            }
        """)
        variables = GraphVariables(table, model)
        str_table = ibis.memtable({"val": ["a", "b"]})
        variables["input"] = str_table["val"]
        with pytest.raises(ValueError, match="numeric"):
            SigmoidTranslator(
                table, model.node[0], variables, self.optimizer, TranslationOptions()
            ).process()


class TestLeakyReluTranslator:
    """Tests for LeakyReluTranslator."""

    def test_leakyrelu_registered(self):
        from orbital.translation.steps.leakyrelu import LeakyReluTranslator

        assert TRANSLATORS.get("LeakyRelu") is LeakyReluTranslator

    def test_leakyrelu_positive(self):
        table = ibis.memtable({"x": [1.0, 2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = LeakyRelu(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.leakyrelu import LeakyReluTranslator

        t = LeakyReluTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [1.0, 2.0]

    def test_leakyrelu_negative_custom_alpha(self):
        """Negative values scaled by custom alpha."""
        table = ibis.memtable({"x": [-2.0, -4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = LeakyRelu <alpha: float = 0.1> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.leakyrelu import LeakyReluTranslator

        t = LeakyReluTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert abs(result[0] - (-0.2)) < 1e-6
        assert abs(result[1] - (-0.4)) < 1e-6


class TestHardSigmoidTranslator:
    """Tests for HardSigmoidTranslator."""

    def test_hardsigmoid_registered(self):
        from orbital.translation.steps.hardsigmoid import HardSigmoidTranslator

        assert TRANSLATORS.get("HardSigmoid") is HardSigmoidTranslator

    def test_hardsigmoid_clips_to_range(self):
        """HardSigmoid clips to [0, 1]."""
        table = ibis.memtable({"x": [-100.0, 0.0, 100.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = HardSigmoid(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.hardsigmoid import HardSigmoidTranslator

        t = HardSigmoidTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result[0] == 0.0
        assert result[2] == 1.0
        assert 0.0 < result[1] < 1.0

    def test_hardsigmoid_midpoint(self):
        """At x=0 with defaults (alpha=0.2, beta=0.5): 0.2*0 + 0.5 = 0.5."""
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = HardSigmoid(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.hardsigmoid import HardSigmoidTranslator

        t = HardSigmoidTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - 0.5) < 1e-9


class TestSoftsignTranslator:
    """Tests for SoftsignTranslator."""

    def test_softsign_registered(self):
        from orbital.translation.steps.softsign import SoftsignTranslator

        assert TRANSLATORS.get("Softsign") is SoftsignTranslator

    def test_softsign_zero(self):
        """Softsign(0) = 0 / (1 + 0) = 0."""
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Softsign(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.softsign import SoftsignTranslator

        t = SoftsignTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        assert backend.execute(variables.peek_variable("output"))[0] == 0.0

    def test_softsign_positive(self):
        """Softsign(3) = 3 / (1 + 3) = 0.75."""
        table = ibis.memtable({"x": [3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Softsign(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.softsign import SoftsignTranslator

        t = SoftsignTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - 0.75) < 1e-9

    def test_softsign_negative(self):
        """Softsign(-4) = -4 / (1 + 4) = -0.8."""
        table = ibis.memtable({"x": [-4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Softsign(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.softsign import SoftsignTranslator

        t = SoftsignTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - (-0.8)) < 1e-9


class TestLogSoftmaxTranslator:
    """Tests for LogSoftmaxTranslator."""

    def test_logsoftmax_registered(self):
        """Verify LogSoftmaxTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.logsoftmax import LogSoftmaxTranslator

        assert TRANSLATORS.get("LogSoftmax") is LogSoftmaxTranslator

    def test_logsoftmax_group(self):
        """LogSoftmax of a column group sums to log(1) per row (all exp sum = 1 → log=0)."""
        import math

        multi_table = ibis.memtable({"a": [1.0], "b": [2.0], "c": [3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = LogSoftmax(input)
            }
        """)
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {"a": multi_table["a"], "b": multi_table["b"], "c": multi_table["c"]}
        )
        from orbital.translation.steps.logsoftmax import LogSoftmaxTranslator

        t = LogSoftmaxTranslator(
            multi_table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        backend = ibis.duckdb.connect()
        vals = [backend.execute(result[k])[0] for k in ["a", "b", "c"]]
        # All log-softmax values must be ≤ 0 and exp(vals) must sum to 1
        assert all(v <= 0 for v in vals)
        assert abs(sum(math.exp(v) for v in vals) - 1.0) < 1e-9

    def test_logsoftmax_single(self):
        """LogSoftmax of a single input is always 0."""
        table = ibis.memtable({"x": [5.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = LogSoftmax(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.logsoftmax import LogSoftmaxTranslator

        t = LogSoftmaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        assert backend.execute(variables.peek_variable("output")) == 0.0


# ---------------------------------------------------------------------------
# Unit tests: SwishTranslator
# ---------------------------------------------------------------------------
