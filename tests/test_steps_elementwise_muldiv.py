"""Tests for binary arithmetic pipeline step translators (Add, Sub, Mul, Div)."""

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
from orbital.translation.steps.softmax import SoftmaxTranslator
from orbital.translation.steps.imputer import ImputerTranslator
from orbital.translation.steps.argmax import ArgMaxTranslator
from orbital.translation.steps.add import AddTranslator
from orbital.translation.steps.sub import SubTranslator
from orbital.translation.steps.mul import MulTranslator
from orbital.translation.steps.div import DivTranslator
from orbital.translation.steps.identity import IdentityTranslator
from orbital.translation.steps.reshape import ReshapeTranslator
from orbital.translation.steps.matmul import MatMulTranslator
from orbital.translation.steps.cast import CastTranslator, CastLikeTranslator
from orbital.translation.steps.linearclass TestMulTranslator:

    def test_mul_single_column(self):
        """Test MulTranslator with a single column input."""
        table = ibis.memtable({"input": [2.0, 3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] mul_value = {5.0}>
            {
                output = Mul(input, mul_value)
            }
        """)

        variables = GraphVariables(table, model)
        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [10.0, 15.0, 20.0]

    def test_mul_group_columns(self):
        """Test MulTranslator with a group of columns."""
        table = ibis.memtable(
            {
                "col_a": [2.0, 3.0, 4.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] mul_values = {3.0, 2.0}>
            {
                output = Mul(input, mul_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [6.0, 9.0, 12.0]
        assert list(backend.execute(result["col_b"])) == [20.0, 40.0, 60.0]

    def test_mul_invalid_non_numeric(self):
        """Test MulTranslator raises error for non-numeric operand."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] mul_value = {5.0}>
            {
                output = Mul(input, mul_value)
            }
        """)

        variables = GraphVariables(table, model)
        variables["input"] = "not_a_numeric_value"  # type: ignore[assignment]

        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="first operand must be a numeric value"):
            translator.process()

    def test_mul_mismatched_column_count(self):
        """Test MulTranslator raises error when column count doesn't match."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] mul_values = {5.0}>
            {
                output = Mul(input, mul_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="same number of values"):
            translator.process()

    def test_mul_single_column_requires_single_value(self):
        """Test MulTranslator raises error when single column given multiple values."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] mul_values = {5.0, 10.0}>
            {
                output = Mul(input, mul_values)
            }
        """)

        variables = GraphVariables(table, model)

        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="must contain exactly 1 value"):
            translator.process()

    def test_mul_var_times_var(self):
        """MulTranslator supports variable × variable (column × column)."""
        table = ibis.memtable(
            {
                "input": [1.0, 2.0, 3.0],
                "other": [5.0, 5.0, 5.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input, float[N] other) => (float[N] output) {
                output = Mul(input, other)
            }
        """)

        variables = GraphVariables(table, model)

        MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [5.0, 10.0, 15.0]



class TestDivTranslator:

    def test_div_single_column(self):
        """Test DivTranslator with a single column input."""
        table = ibis.memtable({"input": [10.0, 20.0, 30.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] div_value = {2.0}>
            {
                output = Div(input, div_value)
            }
        """)

        variables = GraphVariables(table, model)
        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [5.0, 10.0, 15.0]

    def test_div_group_columns_with_matching_values(self):
        """Test DivTranslator with group of columns and matching divisor values."""
        table = ibis.memtable(
            {
                "col_a": [10.0, 20.0, 30.0],
                "col_b": [100.0, 200.0, 300.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] div_values = {2.0, 10.0}>
            {
                output = Div(input, div_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [5.0, 10.0, 15.0]
        assert list(backend.execute(result["col_b"])) == [10.0, 20.0, 30.0]

    def test_div_group_columns_broadcast_single_value(self):
        """Test DivTranslator with group of columns and single divisor (broadcast)."""
        table = ibis.memtable(
            {
                "col_a": [10.0, 20.0, 30.0],
                "col_b": [100.0, 200.0, 300.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] div_value = {10.0}>
            {
                output = Div(input, div_value)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["col_b"])) == [10.0, 20.0, 30.0]

    def test_div_invalid_non_numeric_single(self):
        """Test DivTranslator raises error for non-numeric single operand."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] div_value = {5.0}>
            {
                output = Div(input, div_value)
            }
        """)

        variables = GraphVariables(table, model)
        variables["input"] = "not_a_numeric_value"  # type: ignore[assignment]

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="first operand must be a numeric value"):
            translator.process()

    def test_div_mismatched_column_count(self):
        """Test DivTranslator raises error when column count doesn't match values."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[3] div_values = {5.0, 10.0, 15.0}>
            {
                output = Div(input, div_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="must match the number of columns"):
            translator.process()

    def test_div_single_column_requires_single_value(self):
        """Test DivTranslator raises error when single column given multiple values."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] div_values = {5.0, 10.0}>
            {
                output = Div(input, div_values)
            }
        """)

        variables = GraphVariables(table, model)

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="must contain only one value"):
            translator.process()

    def test_div_var_div_var(self):
        """DivTranslator supports variable ÷ variable (column ÷ column)."""
        table = ibis.memtable(
            {
                "input": [1.0, 2.0, 3.0],
                "other": [5.0, 5.0, 5.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input, float[N] other) => (float[N] output) {
                output = Div(input, other)
            }
        """)

        variables = GraphVariables(table, model)

        DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        for got, expected in zip(result, [0.2, 0.4, 0.6]):
            assert abs(got - expected) < 1e-9
