"""Tests for MatMul pipeline step translators."""

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
from orbital.translation.steps.matmul import MatMulTranslator


from conftest import make_graph_with_inits as _make_graph_with_inits


class TestMatMulTranslator:
    def test_matmul_single_column_times_1d_weight_vector_length_1(self):
        """Test MatMulTranslator with single column times 1D weight vector (coefficient vector of length 1)."""
        table = ibis.memtable({"input": [2.0, 3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] weights = {5.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        variables = GraphVariables(table, model)
        translator = MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [10.0, 15.0, 20.0]

    def test_matmul_group_columns_times_2d_weight_matrix_multiple_outputs(self):
        """Test MatMulTranslator with group of columns times 2D weight matrix (produces multiple outputs)."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
            }
        )
        # Flattened row-major: [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[6] weights = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        model.initializer[0].dims[:] = [2, 3]

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 3

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["out_0"])) == [8.5, 11.0, 13.5]
        assert list(backend.execute(result["out_1"])) == [11.0, 14.5, 18.0]
        assert list(backend.execute(result["out_2"])) == [13.5, 18.0, 22.5]

    def test_matmul_group_columns_times_1d_weight_vector_single_output(self):
        """Test MatMulTranslator with group of columns times 1D weight vector (produces single output)."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
                "col_c": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[3] weights = {2.0, 3.0, 4.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        model.initializer[0].dims[:] = [3]

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
                "col_c": table["col_c"],
            }
        )

        translator = MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [42.0, 51.0, 60.0]

    def test_matmul_with_optimizer_enabled(self):
        """Test MatMulTranslator with optimizer folding enabled."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] weights = {2.0, 3.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        model.initializer[0].dims[:] = [2]

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = MatMulTranslator(
            table,
            model.node[0],
            variables,
            Optimizer(enabled=True),
            TranslationOptions(),
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [14.0, 19.0, 24.0]

    def test_matmul_error_non_numeric_input_columns(self):
        """Test MatMulTranslator raises error for non-numeric input columns."""
        table = ibis.memtable({"input": ["a", "b", "c"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] input) => (float[N] output)
            <float[1] weights = {2.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        model.initializer[0].dims[:] = [1]
        variables = GraphVariables(table, model)

        translator = MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="first operand must be a numeric column"):
            translator.process()
