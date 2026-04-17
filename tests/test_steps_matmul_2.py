"""Tests for MatMul pipeline step translators (split from test_steps_matmul.py)."""

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
    def test_matmul_error_weight_matrix_dimension_mismatch(self):
        """Test MatMulTranslator raises error for weight matrix dimension mismatch."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
            }
        )
        # Weight matrix [3, 2] expects 3 features but input provides only 2
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[6] weights = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        model.initializer[0].dims[:] = [3, 2]

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

        with pytest.raises(ValueError, match="Mismatch: number of features"):
            translator.process()

    def test_matmul_error_unsupported_weight_tensor_rank(self):
        """Test MatMulTranslator raises error for unsupported weight tensor shapes (rank > 2)."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[8] weights = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        model.initializer[0].dims[:] = [2, 2, 2]  # 3D tensor - unsupported

        variables = GraphVariables(table, model)

        translator = MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError, match="coefficient tensor rank > 2 is not supported"
        ):
            translator.process()

    def test_matmul_error_single_column_with_coefficient_vector_longer_than_1(self):
        """Test MatMulTranslator raises error for single column with coefficient vector longer than 1."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[3] weights = {2.0, 3.0, 4.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        # Set weight vector dimensions [3] - but we have single column
        model.initializer[0].dims[:] = [3]

        variables = GraphVariables(table, model)

        translator = MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="Expected coefficient vector of length 1 for single operand",
        ):
            translator.process()

    def test_matmul_group_columns_times_2d_weight_matrix_single_output(self):
        """Test MatMulTranslator with group of columns times 2D weight matrix with single output."""
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

        model.initializer[0].dims[:] = [2, 1]

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
        assert not isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [14.0, 19.0, 24.0]

    def test_matmul_single_column_times_2d_weight_matrix_multiple_outputs(self):
        """Test MatMulTranslator with single column times 2D weight matrix [1, N] (produces multiple outputs)."""
        table = ibis.memtable({"input": [2.0, 3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[3] weights = {2.0, 3.0, 4.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        model.initializer[0].dims[:] = [1, 3]

        variables = GraphVariables(table, model)

        translator = MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 3

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["out_0"])) == [4.0, 6.0, 8.0]
        assert list(backend.execute(result["out_1"])) == [6.0, 9.0, 12.0]
        assert list(backend.execute(result["out_2"])) == [8.0, 12.0, 16.0]

    def test_dynamic_B_raises_NotImplementedError(self):
        """MatMul where B is a dynamic variable (not a constant initializer) raises NotImplementedError."""
        # B is listed as a graph input with no matching initializer → dynamic
        node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["output"])
        graph = _make_graph_with_inits(
            node,
            [
                helper.make_tensor_value_info("A", TensorProto.FLOAT, [None]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [None]),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None])],
            [],  # no initializers — B is dynamic
        )
        table = ibis.memtable({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        variables = GraphVariables(table, graph)

        with pytest.raises(NotImplementedError):
            MatMulTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()

    def test_static_B_still_works(self):
        """MatMul with a static B initializer still produces the correct scalar product."""
        table = ibis.memtable({"input": [3.0, 4.0, 5.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] weights = {2.0}>
            {
                output = MatMul(input, weights)
            }
        """)
        variables = GraphVariables(table, model)
        MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result)) == [6.0, 8.0, 10.0]
