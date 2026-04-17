"""Tests for CumSum GatherND Expand pipeline step translators."""

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

from orbital.translation.steps.cumsum import CumSumTranslator
from orbital.translation.steps.gathernd import GatherNDTranslator
from orbital.translation.steps.expand import ExpandTranslator

from conftest import make_graph_with_inits as _make_graph_with_inits


class TestCumSumTranslator:
    def test_cumsum_inclusive_prefix(self):
        """CumSum exclusive=0, reverse=0: inclusive prefix sum."""
        table = ibis.memtable({"a": [1.0], "b": [2.0], "c": [3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <int64[1] axis = {1}>
            {
                output = CumSum <exclusive: int = 0, reverse: int = 0> (x, axis)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup(
            {"a": table["a"], "b": table["b"], "c": table["c"]}
        )

        translator = CumSumTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["a"])) == [1.0]
        assert list(backend.execute(result["b"])) == [3.0]
        assert list(backend.execute(result["c"])) == [6.0]

    def test_cumsum_exclusive_prefix(self):
        """CumSum exclusive=1, reverse=0: exclusive prefix sum (first=0)."""
        table = ibis.memtable({"a": [1.0], "b": [2.0], "c": [3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <int64[1] axis = {1}>
            {
                output = CumSum <exclusive: int = 1, reverse: int = 0> (x, axis)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup(
            {"a": table["a"], "b": table["b"], "c": table["c"]}
        )

        translator = CumSumTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["a"])) == [0]
        assert list(backend.execute(result["b"])) == [1.0]
        assert list(backend.execute(result["c"])) == [3.0]

    def test_cumsum_suffix(self):
        """CumSum exclusive=0, reverse=1: inclusive suffix sum."""
        table = ibis.memtable({"a": [1.0], "b": [2.0], "c": [3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <int64[1] axis = {1}>
            {
                output = CumSum <exclusive: int = 0, reverse: int = 1> (x, axis)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup(
            {"a": table["a"], "b": table["b"], "c": table["c"]}
        )

        translator = CumSumTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["a"])) == [6.0]
        assert list(backend.execute(result["b"])) == [5.0]
        assert list(backend.execute(result["c"])) == [3.0]


# ---------------------------------------------------------------------------
# GatherND tests
# ---------------------------------------------------------------------------


class TestGatherNDTranslator:
    def test_gathernd_basic(self):
        """GatherND selects column by integer index per row."""
        table = ibis.memtable(
            {
                "col_0": [10.0, 20.0, 30.0],
                "col_1": [40.0, 50.0, 60.0],
                "col_2": [70.0, 80.0, 90.0],
                "indices": [0, 2, 1],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data, int64[N] indices) => (float[N] output) {
                output = GatherND (data, indices)
            }
        """)
        # GraphVariables needs both formal input columns in its init table.
        init_table = ibis.memtable({"data": [1.0], "indices": [0]})
        variables = GraphVariables(init_table, model)
        variables["data"] = NumericVariablesGroup(
            {
                "col_0": table["col_0"],
                "col_1": table["col_1"],
                "col_2": table["col_2"],
            }
        )
        variables["indices"] = table["indices"]

        translator = GatherNDTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        # Row 0: idx=0 → col_0=10, Row 1: idx=2 → col_2=80, Row 2: idx=1 → col_1=60
        assert list(backend.execute(result)) == [10.0, 80.0, 60.0]

    def test_gathernd_batch_dims_nonzero_raises(self):
        """GatherND with batch_dims != 0 raises NotImplementedError."""
        table = ibis.memtable({"col_0": [1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data, int64[N] indices) => (float[N] output) {
                output = GatherND <batch_dims: int = 1> (data, indices)
            }
        """)
        init_table = ibis.memtable({"data": [1.0], "indices": [0]})
        variables = GraphVariables(init_table, model)
        variables["data"] = NumericVariablesGroup({"col_0": table["col_0"]})

        translator = GatherNDTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        with pytest.raises(NotImplementedError, match="batch_dims=0"):
            translator.process()


# ---------------------------------------------------------------------------
# Expand tests
# ---------------------------------------------------------------------------


class TestExpandTranslator:
    def test_expand_scalar_to_n_columns(self):
        """Expand replicates a single column to N columns."""
        table = ibis.memtable({"val": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <int64[1] shape = {3}>
            {
                output = Expand (input, shape)
            }
        """)
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = table["val"]

        translator = ExpandTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 3

        backend = ibis.duckdb.connect()
        for i in range(3):
            assert list(backend.execute(result[i])) == [1.0, 2.0, 3.0]

    def test_expand_single_column_dict_to_n(self):
        """Expand works when input is a single-entry dict."""
        table = ibis.memtable({"val": [5.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <int64[1] shape = {2}>
            {
                output = Expand (input, shape)
            }
        """)
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup({"val": table["val"]})

        translator = ExpandTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result[0])) == [5.0]
        assert list(backend.execute(result[1])) == [5.0]
