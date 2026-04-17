"""Tests for arithmetic pipeline step translators."""

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


class TestAbsTranslator:
    """Tests for AbsTranslator."""

    def test_abs_registered(self):
        from orbital.translation.steps.abs import AbsTranslator

        assert TRANSLATORS.get("Abs") is AbsTranslator

    def test_abs_single_column(self):
        table = ibis.memtable({"x": [-3.0, 0.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Abs(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.abs import AbsTranslator

        t = AbsTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [
            3.0,
            0.0,
            4.0,
        ]

    def test_abs_group(self):
        table = ibis.memtable({"a": [-1.0, 2.0], "b": [3.0, -4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Abs(x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.abs import AbsTranslator

        t = AbsTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["a"])) == [1.0, 2.0]
        assert list(backend.execute(result["b"])) == [3.0, 4.0]


class TestSqrtTranslator:
    """Tests for SqrtTranslator."""

    def test_sqrt_registered(self):
        from orbital.translation.steps.sqrt import SqrtTranslator

        assert TRANSLATORS.get("Sqrt") is SqrtTranslator

    def test_sqrt_single_column(self):
        table = ibis.memtable({"x": [4.0, 9.0, 16.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Sqrt(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.sqrt import SqrtTranslator

        t = SqrtTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [
            2.0,
            3.0,
            4.0,
        ]

    def test_sqrt_group(self):
        table = ibis.memtable({"a": [1.0, 4.0], "b": [9.0, 16.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Sqrt(x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.sqrt import SqrtTranslator

        t = SqrtTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["a"])) == [1.0, 2.0]
        assert list(backend.execute(result["b"])) == [3.0, 4.0]


class TestErfTranslator:
    """Tests for ErfTranslator."""

    def test_erf_registered(self):
        from orbital.translation.steps.erf import ErfTranslator

        assert TRANSLATORS.get("Erf") is ErfTranslator

    def test_erf_zero_input(self):
        """erf(0) == 0."""
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Erf(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.erf import ErfTranslator

        t = ErfTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - 0.0) < 1e-9

    def test_erf_large_positive(self):
        """erf(large) ≈ 1.0."""
        table = ibis.memtable({"x": [5.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Erf(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.erf import ErfTranslator

        t = ErfTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - 1.0) < 1e-6


class TestExpTranslator:
    """Tests for the ONNX Exp translator (e^x)."""

    def test_exp_registered(self):
        from orbital.translation.steps.exp import ExpTranslator

        assert TRANSLATORS.get("Exp") is ExpTranslator

    def test_exp_single_column(self):
        """Exp of a single column computes e^x per row."""
        import math

        table = ibis.memtable({"x": [0.0, 1.0, -1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Exp(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.exp import ExpTranslator

        ExpTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        expected = [math.exp(v) for v in [0.0, 1.0, -1.0]]
        for got, exp in zip(result, expected):
            assert abs(got - exp) < 1e-9

    def test_exp_group_of_columns(self):
        """Exp applied element-wise to a column group."""
        import math

        table = ibis.memtable({"a": [0.0, 2.0], "b": [1.0, -1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Exp(x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.exp import ExpTranslator

        ExpTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("output")
        assert isinstance(result, VariablesGroup)
        backend = ibis.duckdb.connect()
        a_vals = list(backend.execute(result["a"]))
        b_vals = list(backend.execute(result["b"]))
        for got, raw in zip(a_vals, [0.0, 2.0]):
            assert abs(got - math.exp(raw)) < 1e-9
        for got, raw in zip(b_vals, [1.0, -1.0]):
            assert abs(got - math.exp(raw)) < 1e-9


# ---------------------------------------------------------------------------
# Unit tests: SwishTranslator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit tests: LogTranslator
# ---------------------------------------------------------------------------


class TestLogTranslator:
    """Tests for the ONNX Log translator (ln(x))."""

    def test_log_registered(self):
        from orbital.translation.steps.log import LogTranslator

        assert TRANSLATORS.get("Log") is LogTranslator

    def test_log_single_column(self):
        """Log of a single column computes ln(x) per row."""
        import math

        table = ibis.memtable({"x": [1.0, math.e, math.e**2]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Log(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.log import LogTranslator

        LogTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        expected = [math.log(v) for v in [1.0, math.e, math.e**2]]
        for got, exp in zip(result, expected):
            assert abs(got - exp) < 1e-9

    def test_log_group_of_columns(self):
        """Log applied element-wise to a column group."""
        import math

        table = ibis.memtable({"a": [1.0, math.e], "b": [math.e**2, math.e**3]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Log(x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.log import LogTranslator

        LogTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("output")
        assert isinstance(result, VariablesGroup)
        backend = ibis.duckdb.connect()
        a_vals = list(backend.execute(result["a"]))
        b_vals = list(backend.execute(result["b"]))
        for got, raw in zip(a_vals, [1.0, math.e]):
            assert abs(got - math.log(raw)) < 1e-9
        for got, raw in zip(b_vals, [math.e**2, math.e**3]):
            assert abs(got - math.log(raw)) < 1e-9
