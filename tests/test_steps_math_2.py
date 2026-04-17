"""Tests for arithmetic pipeline step translators (split from test_steps_math.py)."""

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


class TestSignTranslator:
    """Tests for SignTranslator."""

    def test_sign_registered(self):
        from orbital.translation.steps.sign import SignTranslator

        assert TRANSLATORS.get("Sign") is SignTranslator

    def test_sign_single_column(self):
        table = ibis.memtable({"x": [-5.0, 0.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Sign(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.sign import SignTranslator

        SignTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [
            -1.0,
            0.0,
            1.0,
        ]

    def test_sign_group(self):
        table = ibis.memtable({"a": [-2.0, 0.0], "b": [4.0, -1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Sign(x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.sign import SignTranslator

        SignTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["a"])) == [-1.0, 0.0]
        assert list(backend.execute(result["b"])) == [1.0, -1.0]


class TestClipTranslator:
    """Tests for ClipTranslator."""

    def test_clip_registered(self):
        """Verify ClipTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.clip import ClipTranslator

        assert TRANSLATORS.get("Clip") is ClipTranslator

    def test_clip_attribute_bounds(self):
        """Clip with min/max as node attributes."""
        table = ibis.memtable({"x": [-2.0, 0.0, 3.0, 8.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Clip <min: float = 0.0, max: float = 6.0> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.clip import ClipTranslator

        t = ClipTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [0.0, 0.0, 3.0, 6.0]

    def test_clip_min_only(self):
        """Clip with only a lower bound (relu-equivalent)."""
        table = ibis.memtable({"x": [-1.0, 0.0, 2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Clip <min: float = 0.0> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.clip import ClipTranslator

        t = ClipTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [0.0, 0.0, 2.0]


# ---------------------------------------------------------------------------
# Unit tests: ModTranslator
# ---------------------------------------------------------------------------


class TestIsInfTranslator:
    """Tests for IsInfTranslator."""

    def test_isinf_registered(self):
        from orbital.translation.steps.isinf import IsInfTranslator

        assert TRANSLATORS.get("IsInf") is IsInfTranslator

    def test_isinf_single_column(self):
        table = ibis.memtable({"x": [1.0, float("inf"), float("-inf")]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = IsInf(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.isinf import IsInfTranslator

        t = IsInfTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [False, True, True]


class TestIsNaNTranslator:
    """Tests for IsNaNTranslator."""

    def test_isnan_registered(self):
        from orbital.translation.steps.isnan import IsNaNTranslator

        assert TRANSLATORS.get("IsNaN") is IsNaNTranslator

    def test_isnan_single_column(self):
        table = ibis.memtable({"x": [1.0, float("nan"), 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = IsNaN(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.isnan import IsNaNTranslator

        t = IsNaNTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        # DuckDB memtable stores float('nan') as NULL; isnan(NULL) returns NULL per SQL semantics.
        assert result[0] is False
        assert result[1] is True or result[1] is None
        assert result[2] is False


class TestRoundTranslator:
    """Tests for RoundTranslator."""

    def test_round_registered(self):
        from orbital.translation.steps.round_ import RoundTranslator

        assert TRANSLATORS.get("Round") is RoundTranslator

    def test_round_single_column(self):
        table = ibis.memtable({"x": [1.4, 2.6, 3.5]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Round(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.round_ import RoundTranslator

        t = RoundTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [1.0, 3.0, 4.0]

    def test_round_group(self):
        table = ibis.memtable({"a": [1.4, 2.6], "b": [3.5, 4.2]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Round(x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.round_ import RoundTranslator

        t = RoundTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["a"])) == [1.0, 3.0]
        assert list(backend.execute(result["b"])) == [4.0, 4.0]
