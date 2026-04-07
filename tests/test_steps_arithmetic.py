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
from orbital.translation.steps.linearclass import LinearClassifierTranslator
from orbital.translation.steps.linearreg import LinearRegressorTranslator
from orbital.translation.steps.scaler import ScalerTranslator
from orbital.translation.steps.onehotencoder import OneHotEncoderTranslator
from orbital.translation.steps.labelencoder import LabelEncoderTranslator
from orbital.translation.steps.where import WhereTranslator
from orbital.translation.steps.zipmap import ZipMapTranslator
from orbital.translation.steps.concat import ConcatTranslator
from orbital.translation.steps.featurevectorizer import FeatureVectorizerTranslator
from orbital.translation.steps.gather import GatherTranslator
from orbital.translation.steps.arrayfeatureextractor import ArrayFeatureExtractorTranslator



# ---------------------------------------------------------------------------
# Helper: build an ONNX graph with weight initializers
# ---------------------------------------------------------------------------

def _make_graph_with_inits(node, inputs_info, outputs_info, initializers):
    '''Create an ONNX GraphProto with given node, I/O specs, and initializers.'''
    return helper.make_graph(
        [node],
        "test_graph",
        inputs_info,
        outputs_info,
        initializer=initializers,
    )

class TestAddTranslator:

    def test_add_single_column(self):
        """Test AddTranslator with a single column input."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] add_value = {5.0}>
            {
                output = Add(input, add_value)
            }
        """)

        variables = GraphVariables(table, model)
        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [6.0, 7.0, 8.0]

    def test_add_group_columns(self):
        """Test AddTranslator with a group of columns."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] add_values = {5.0, 100.0}>
            {
                output = Add(input, add_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [6.0, 7.0, 8.0]
        assert list(backend.execute(result["col_b"])) == [110.0, 120.0, 130.0]

    def test_add_invalid_non_numeric(self):
        """Test AddTranslator raises error for non-numeric operand."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] add_value = {5.0}>
            {
                output = Add(input, add_value)
            }
        """)

        variables = GraphVariables(table, model)
        variables["input"] = "not_a_numeric_value"  # type: ignore[assignment]

        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="first operand must be a numeric value"):
            translator.process()

    def test_add_mismatched_column_count(self):
        """Test AddTranslator raises error when column count doesn't match."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] add_values = {5.0}>
            {
                output = Add(input, add_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="same number of values"):
            translator.process()

    def test_add_single_column_requires_single_value(self):
        """Test AddTranslator raises error when single column given multiple values."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] add_values = {5.0, 10.0}>
            {
                output = Add(input, add_values)
            }
        """)

        variables = GraphVariables(table, model)

        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="must contain exactly 1 value"):
            translator.process()

    def test_add_second_operand_variable(self):
        """Test AddTranslator supports variable-to-variable addition (residual connections)."""
        table = ibis.memtable(
            {
                "input": [1.0, 2.0, 3.0],
                "other": [5.0, 5.0, 5.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input, float[N] other) => (float[N] output) {
                output = Add(input, other)
            }
        """)

        variables = GraphVariables(table, model)

        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        values = backend.execute(result).tolist()
        assert values == [6.0, 7.0, 8.0]


class TestSubTranslator:

    def test_sub_single_column(self):
        """Test SubTranslator with a single column input."""
        table = ibis.memtable({"input": [10.0, 20.0, 30.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] sub_value = {3.0}>
            {
                output = Sub(input, sub_value)
            }
        """)

        variables = GraphVariables(table, model)
        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [7.0, 17.0, 27.0]

    def test_sub_group_columns(self):
        """Test SubTranslator with a group of columns."""
        table = ibis.memtable(
            {
                "col_a": [10.0, 20.0, 30.0],
                "col_b": [100.0, 200.0, 300.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] sub_values = {3.0, 50.0}>
            {
                output = Sub(input, sub_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [7.0, 17.0, 27.0]
        assert list(backend.execute(result["col_b"])) == [50.0, 150.0, 250.0]

    def test_sub_invalid_non_numeric(self):
        """Test SubTranslator raises error for non-numeric operand."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] sub_value = {5.0}>
            {
                output = Sub(input, sub_value)
            }
        """)

        variables = GraphVariables(table, model)
        variables["input"] = "not_a_numeric_value"  # type: ignore[assignment]

        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="first operand must be a numeric value"):
            translator.process()

    def test_sub_single_column_requires_single_value(self):
        """Test SubTranslator raises error when single column given multiple values."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] sub_values = {5.0, 10.0}>
            {
                output = Sub(input, sub_values)
            }
        """)

        variables = GraphVariables(table, model)

        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="must contain exactly 1 value"):
            translator.process()

    def test_sub_mismatched_column_count(self):
        """Test SubTranslator raises error when column count doesn't match."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] sub_values = {5.0}>
            {
                output = Sub(input, sub_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="variable group has"):
            translator.process()

    def test_sub_variable_minus_variable(self):
        """Test SubTranslator supports variable-to-variable subtraction."""
        table = ibis.memtable(
            {
                "input": [10.0, 20.0, 30.0],
                "other": [5.0, 5.0, 5.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input, float[N] other) => (float[N] output) {
                output = Sub(input, other)
            }
        """)

        variables = GraphVariables(table, model)

        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        values = backend.execute(result).tolist()
        assert values == [5.0, 15.0, 25.0]


class TestMulTranslator:

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


class TestNegTranslator:
    """Tests for NegTranslator."""


    def test_neg_registered(self):
        from orbital.translation.steps.neg import NegTranslator
        assert TRANSLATORS.get("Neg") is NegTranslator

    def test_neg_single_column(self):
        table = ibis.memtable({"x": [1.0, -2.0, 0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Neg(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.neg import NegTranslator
        t = NegTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [-1.0, 2.0, 0.0]

    def test_neg_group(self):
        table = ibis.memtable({"a": [1.0, -2.0], "b": [-3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Neg(x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.neg import NegTranslator
        t = NegTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["a"])) == [-1.0, 2.0]
        assert list(backend.execute(result["b"])) == [3.0, -4.0]


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
        t = AbsTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [3.0, 0.0, 4.0]

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
        t = AbsTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
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
        t = SqrtTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [2.0, 3.0, 4.0]

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
        t = SqrtTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
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
        t = ErfTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
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
        t = ErfTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - 1.0) < 1e-6


class TestPowTranslator:
    """Tests for PowTranslator."""


    def test_pow_registered(self):
        from orbital.translation.steps.pow import PowTranslator
        assert TRANSLATORS.get("Pow") is PowTranslator

    def test_pow_single_column_constant_exp(self):
        """x^3 with constant exponent from initializer."""
        table = ibis.memtable({"x": [2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <float[1] exp = {3.0}>
            {
                output = Pow(x, exp)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.pow import PowTranslator
        t = PowTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [8.0, 27.0]

    def test_pow_group_constant_exp(self):
        """Group raised to constant exponent."""
        table = ibis.memtable({"a": [2.0, 3.0], "b": [4.0, 5.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <float[1] exp = {2.0}>
            {
                output = Pow(x, exp)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.pow import PowTranslator
        t = PowTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["a"])) == [4.0, 9.0]
        assert list(backend.execute(result["b"])) == [16.0, 25.0]


# ---------------------------------------------------------------------------
# Unit tests: ExpTranslator
# ---------------------------------------------------------------------------


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
        table = ibis.memtable({"x": [1.0, math.e, math.e ** 2]})
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
        expected = [math.log(v) for v in [1.0, math.e, math.e ** 2]]
        for got, exp in zip(result, expected):
            assert abs(got - exp) < 1e-9

    def test_log_group_of_columns(self):
        """Log applied element-wise to a column group."""
        import math
        table = ibis.memtable({"a": [1.0, math.e], "b": [math.e ** 2, math.e ** 3]})
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
        for got, raw in zip(b_vals, [math.e ** 2, math.e ** 3]):
            assert abs(got - math.log(raw)) < 1e-9


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
        assert list(backend.execute(variables.peek_variable("output"))) == [-1.0, 0.0, 1.0]

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
        t = ClipTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
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
        t = ClipTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [0.0, 0.0, 2.0]


# ---------------------------------------------------------------------------
# Unit tests: ModTranslator
# ---------------------------------------------------------------------------


class TestModTranslator:
    """Tests for ModTranslator."""


    def test_mod_registered(self):
        from orbital.translation.steps.mod import ModTranslator
        assert TRANSLATORS.get("Mod") is ModTranslator

    def test_mod_positive(self):
        """7 % 3 = 1."""
        divisor_tensor = helper.make_tensor("d", TensorProto.FLOAT, [1], [3.0])
        node = helper.make_node("Mod", inputs=["x", "d"], outputs=["y"], fmod=0)
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            [divisor_tensor],
        )
        table = ibis.memtable({"x": [7.0]})
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = table["x"]
        from orbital.translation.steps.mod import ModTranslator
        ModTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("y")
        backend = ibis.duckdb.connect()
        assert abs(list(backend.execute(result))[0] - 1.0) < 1e-6

    def test_fmod_negative(self):
        """fmod(-7, 3): remainder has sign of dividend → -1."""
        divisor_tensor = helper.make_tensor("d", TensorProto.FLOAT, [1], [3.0])
        node = helper.make_node("Mod", inputs=["x", "d"], outputs=["y"], fmod=1)
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            [divisor_tensor],
        )
        table = ibis.memtable({"x": [-7.0]})
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = table["x"]
        from orbital.translation.steps.mod import ModTranslator
        ModTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("y")
        backend = ibis.duckdb.connect()
        val = list(backend.execute(result))[0]
        assert abs(val - (-1.0)) < 1e-6


# ---------------------------------------------------------------------------
# Unit tests: ScatterElementsTranslator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit tests: PadTranslator
# ---------------------------------------------------------------------------


class TestPadTranslator:
    """Tests for PadTranslator (constant padding on feature axis)."""


    def test_pad_registered(self):
        from orbital.translation.steps.pad import PadTranslator
        assert TRANSLATORS.get("Pad") is PadTranslator

    def test_pad_rank1_adds_zero_columns(self):
        """pads=[1, 2] on a rank-1 vector [3.0, 4.0]: [0, 3, 4, 0, 0]."""
        table = ibis.memtable({"a": [3.0], "b": [4.0]})
        pads_tensor = helper.make_tensor("pads", TensorProto.INT64, [2], [1, 2])
        node = helper.make_node("Pad", inputs=["x", "pads"], outputs=["y"])
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            [pads_tensor],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.pad import PadTranslator
        PadTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("y")
        backend = ibis.duckdb.connect()
        # 1 zero before + 2 original columns + 2 zeros after = 5 columns total
        assert len(result) == 5
        assert float(backend.execute(result["pad_begin_0"])) == 0.0
        assert list(backend.execute(result["data_0"]))[0] == 3.0
        assert list(backend.execute(result["data_1"]))[0] == 4.0
        assert float(backend.execute(result["pad_end_0"])) == 0.0
        assert float(backend.execute(result["pad_end_1"])) == 0.0

    def test_pad_rank2_batch_dim_must_be_zero(self):
        """pads=[1, 0, 0, 0] pads the batch dim — must raise NotImplementedError."""
        table = ibis.memtable({"a": [1.0]})
        pads_tensor = helper.make_tensor("pads", TensorProto.INT64, [4], [1, 0, 0, 0])
        node = helper.make_node("Pad", inputs=["x", "pads"], outputs=["y"])
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            [pads_tensor],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = table["a"]
        from orbital.translation.steps.pad import PadTranslator
        with pytest.raises(NotImplementedError, match="batch"):
            PadTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()

    def test_pad_mode_reflect_raises(self):
        """mode='reflect' should raise NotImplementedError."""
        table = ibis.memtable({"a": [1.0]})
        pads_tensor = helper.make_tensor("pads", TensorProto.INT64, [2], [1, 1])
        node = helper.make_node(
            "Pad", inputs=["x", "pads"], outputs=["y"], mode="reflect"
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            [pads_tensor],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = table["a"]
        from orbital.translation.steps.pad import PadTranslator
        with pytest.raises(NotImplementedError, match="reflect"):
            PadTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()


