"""Tests for ArgMin and TopK pipeline step translators."""

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

from orbital.translation.steps.argmin import ArgMinTranslator
from orbital.translation.steps.topk import TopKTranslator

from conftest import make_graph_with_inits as _make_graph_with_inits


class TestArgMinTranslator:
    def test_argmin_basic(self):
        """ArgMin of [3, 1, 2] should return index 1 (position of minimum)."""
        table = ibis.memtable(
            {
                "class_0": [3.0],
                "class_1": [1.0],
                "class_2": [2.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMin <axis: int = 1, keepdims: int = 1> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "class_0": table["class_0"],
                "class_1": table["class_1"],
                "class_2": table["class_2"],
            }
        )

        translator = ArgMinTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result)) == [1]

    def test_argmin_multiple_rows(self):
        """ArgMin with multiple rows selects correct per-row minimum column index."""
        table = ibis.memtable(
            {
                "class_0": [1.0, 5.0, 2.0],
                "class_1": [3.0, 2.0, 8.0],
                "class_2": [2.0, 1.0, 3.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMin <axis: int = 1, keepdims: int = 1> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "class_0": table["class_0"],
                "class_1": table["class_1"],
                "class_2": table["class_2"],
            }
        )

        translator = ArgMinTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        # Row 0: min is class_0 (1.0) -> index 0
        # Row 1: min is class_2 (1.0) -> index 2
        # Row 2: min is class_0 (2.0) -> index 0
        assert computed == [0, 2, 0]

    def test_argmin_select_last_index_ties(self):
        """select_last_index=1 returns last tied minimum column index."""
        table = ibis.memtable(
            {
                "col_0": [1.0],
                "col_1": [1.0],
                "col_2": [2.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMin <axis: int = 1, keepdims: int = 1, select_last_index: int = 1> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "col_0": table["col_0"],
                "col_1": table["col_1"],
                "col_2": table["col_2"],
            }
        )

        translator = ArgMinTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        # col_0 and col_1 are both 1.0 (minimum); select_last_index picks index 1
        assert list(backend.execute(result)) == [1]

    def test_argmin_not_a_group_raises(self):
        """ArgMinTranslator raises NotImplementedError for non-dict input."""
        table = ibis.memtable({"data": [1.0, 2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMin <axis: int = 1> (data)
            }
        """)
        variables = GraphVariables(table, model)
        translator = ArgMinTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        with pytest.raises(NotImplementedError, match="group of columns"):
            translator.process()


# ---------------------------------------------------------------------------
# TopK tests
# ---------------------------------------------------------------------------


class TestTopKTranslator:
    def test_topk_k1_largest(self):
        """TopK with K=1 returns the largest value and its index."""
        table = ibis.memtable(
            {
                "col_0": [1.0, 5.0],
                "col_1": [3.0, 2.0],
                "col_2": [2.0, 1.0],
            }
        )
        # k declared as an ONNX initializer so GraphVariables won't seek it in the table.
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] values, int64[N] indices)
            <int64[1] k = {1}>
            {
                values, indices = TopK <axis: int = 1, largest: int = 1> (data, k)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "col_0": table["col_0"],
                "col_1": table["col_1"],
                "col_2": table["col_2"],
            }
        )

        translator = TopKTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        backend = ibis.duckdb.connect()
        val_group = variables.peek_variable("values")
        idx_group = variables.peek_variable("indices")

        # Row 0: max is col_1 (3.0) at index 1
        # Row 1: max is col_0 (5.0) at index 0
        assert list(backend.execute(val_group[0])) == [3.0, 5.0]
        assert list(backend.execute(idx_group[0])) == [1, 0]

    def test_topk_k2_largest(self):
        """TopK with K=2 returns two largest values and their indices."""
        table = ibis.memtable(
            {
                "col_0": [1.0],
                "col_1": [3.0],
                "col_2": [2.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] values, int64[N] indices)
            <int64[1] k = {2}>
            {
                values, indices = TopK <axis: int = 1, largest: int = 1> (data, k)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "col_0": table["col_0"],
                "col_1": table["col_1"],
                "col_2": table["col_2"],
            }
        )

        translator = TopKTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        backend = ibis.duckdb.connect()
        val_group = variables.peek_variable("values")
        idx_group = variables.peek_variable("indices")

        # Rank-0 (largest): col_1 = 3.0 at index 1
        # Rank-1 (second):  col_2 = 2.0 at index 2
        assert list(backend.execute(val_group[0])) == [3.0]
        assert list(backend.execute(idx_group[0])) == [1]
        assert list(backend.execute(val_group[1])) == [2.0]
        assert list(backend.execute(idx_group[1])) == [2]

    def test_topk_axis_not_1_raises(self):
        """TopK with axis != 1 raises NotImplementedError."""
        table = ibis.memtable({"col_0": [1.0], "col_1": [2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] values, int64[N] indices)
            <int64[1] k = {1}>
            {
                values, indices = TopK <axis: int = 0> (data, k)
            }
        """)
        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {"col_0": table["col_0"], "col_1": table["col_1"]}
        )
        translator = TopKTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        with pytest.raises(NotImplementedError, match="axis=1"):
            translator.process()


# ---------------------------------------------------------------------------
# CumSum tests
# ---------------------------------------------------------------------------
