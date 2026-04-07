"""Tests for ml pipeline step translators."""

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


class TestImputerTranslator:

    def test_imputer_single_column(self):
        """Test ImputerTranslator with a single column input."""
        table = ibis.memtable({"input": [1.0, None, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Imputer <imputed_value_floats: floats = [2.0]> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ImputerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = backend.execute(result)
        assert list(computed) == [1.0, 2.0, 3.0]

    def test_imputer_group_columns(self):
        """Test ImputerTranslator with a group of columns."""
        table = ibis.memtable(
            {
                "col_a": [1.0, None, 3.0],
                "col_b": [None, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Imputer <imputed_value_floats: floats = [10.0, 20.0]> (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = ImputerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [1.0, 10.0, 3.0]
        assert list(backend.execute(result["col_b"])) == [20.0, 5.0, 6.0]

    def test_imputer_invalid_imputed_value_type(self):
        """Test ImputerTranslator raises error for non-list imputed values."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Imputer <imputed_value_floats: floats = [2.0]> (input)
            }
        """)

        variables = GraphVariables(table, model)

        # Override attributes to test validation
        node = model.node[0]
        translator = ImputerTranslator(
            table, node, variables, self.optimizer, TranslationOptions()
        )
        translator._attributes["imputed_value_floats"] = 2.0  # Invalid: not a list

        with pytest.raises(ValueError, match="imputed_value must be a list or tuple"):
            translator.process()

    def test_imputer_mismatched_column_count(self):
        """Test ImputerTranslator raises error when column count doesn't match."""
        table = ibis.memtable(
            {
                "col_a": [1.0, None, 3.0],
                "col_b": [None, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Imputer <imputed_value_floats: floats = [10.0]> (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = ImputerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="number of imputed values does not match"):
            translator.process()


class TestCastTranslator:

    def test_cast_single_column_to_float32(self):
        """Test CastTranslator with single column cast to float32."""
        table = ibis.memtable({"input": [1, 2, 3]})  # int values
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input) => (float[N] output) {
                output = Cast <to: int = 1> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = CastTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = backend.execute(result)
        assert list(computed) == [1.0, 2.0, 3.0]

    def test_cast_single_column_to_float64(self):
        """Test CastTranslator with single column cast to float64."""
        table = ibis.memtable({"input": [1, 2, 3]})  # int values
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input) => (double[N] output) {
                output = Cast <to: int = 11> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = CastTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = backend.execute(result)
        assert list(computed) == [1.0, 2.0, 3.0]

    def test_cast_single_column_to_int64(self):
        """Test CastTranslator with single column cast to int64."""
        table = ibis.memtable({"input": [1.2, 2.7, 3.9]})  # float values
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] output) {
                output = Cast <to: int = 7> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = CastTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = backend.execute(result)
        assert list(computed) == [1, 3, 4]

    def test_cast_single_column_to_string(self):
        """Test CastTranslator with single column cast to string."""
        table = ibis.memtable({"input": [1, 2, 3]})
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input) => (string[N] output) {
                output = Cast <to: int = 8> (input)
            }
        """)

        variables = GraphVariables(table, model)
        options = TranslationOptions(allow_text_tensors=True)
        translator = CastTranslator(
            table, model.node[0], variables, self.optimizer, options
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = backend.execute(result)
        assert list(computed) == ["1", "2", "3"]

    def test_cast_group_of_columns(self):
        """Test CastTranslator with a group of columns."""
        table = ibis.memtable(
            {
                "col_a": [1, 2, 3],
                "col_b": [4, 5, 6],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input) => (float[N] output) {
                output = Cast <to: int = 1> (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = CastTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["col_b"])) == [4.0, 5.0, 6.0]

    def test_skip_cast_to_string_single_column(self):
        """Test CastTranslator skips cast to string when allow_text_tensors=False."""
        table = ibis.memtable({"input": [1, 2, 3]})
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input) => (string[N] output) {
                output = Cast <to: int = 8> (input)
            }
        """)

        variables = GraphVariables(table, model)
        options = TranslationOptions(allow_text_tensors=False)
        translator = CastTranslator(
            table, model.node[0], variables, self.optimizer, options
        )
        translator.process()

        # Should skip the cast and return the original input
        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = backend.execute(result)
        # Should remain int64, not cast to string
        assert list(computed) == [1, 2, 3]

    def test_skip_cast_to_string_column_group(self):
        """Test CastTranslator skips cast to string for column group when allow_text_tensors=False."""
        table = ibis.memtable(
            {
                "col_a": [1, 2, 3],
                "col_b": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (string[N] output) {
                output = Cast <to: int = 8> (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        options = TranslationOptions(allow_text_tensors=False)
        translator = CastTranslator(
            table, model.node[0], variables, self.optimizer, options
        )
        translator.process()

        # Should skip the cast and return the original input group
        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        # Should remain original types, not cast to string
        assert list(backend.execute(result["col_a"])) == [1, 2, 3]
        assert list(backend.execute(result["col_b"])) == [4.0, 5.0, 6.0]

    def test_cast_unsupported_target_type(self):
        """Test CastTranslator raises error for unsupported target type."""
        table = ibis.memtable({"input": [1, 2, 3]})
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input) => (float[N] output) {
                output = Cast <to: int = 999> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = CastTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(NotImplementedError, match="Cast: type 999 not supported"):
            translator.process()


class TestCastLikeTranslator:

    def test_cast_group_to_match_single_column_type(self):
        """Test CastLikeTranslator casts group of columns to match single column type."""
        table = ibis.memtable(
            {
                "col_a": [1, 2, 3],  # int64
                "col_b": [4, 5, 6],  # int64
                "target": [1.0, 2.0, 3.0],  # float64
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input, float[N] target_type) => (float[N] output) {
                output = CastLike(input, target_type)
            }
        """)

        variables = GraphVariables(
            ibis.memtable({"input": [1], "target_type": [1.0]}), model
        )
        variables["input"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )
        variables["target_type"] = table["target"]

        translator = CastLikeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        # Should match target column type (float64)
        assert list(backend.execute(result["col_a"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["col_b"])) == [4.0, 5.0, 6.0]

    def test_castlike_error_input_is_single_column(self):
        """Test CastLikeTranslator raises error when input is single column."""
        table = ibis.memtable(
            {
                "input": [1, 2, 3],
                "target_type": [1.0, 2.0, 3.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input, float[N] target_type) => (float[N] output) {
                output = CastLike(input, target_type)
            }
        """)

        variables = GraphVariables(table, model)

        translator = CastLikeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError,
            match="CastLike currently only supports casting a group of columns",
        ):
            translator.process()

    def test_castlike_error_target_is_group(self):
        """Test CastLikeTranslator raises error when target is a group."""
        table = ibis.memtable(
            {
                "col_a": [1, 2, 3],
                "col_b": [4, 5, 6],
                "target_a": [1.0, 2.0, 3.0],
                "target_b": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input, float[N] target_type) => (float[N] output) {
                output = CastLike(input, target_type)
            }
        """)

        variables = GraphVariables(
            ibis.memtable({"input": [1], "target_type": [1.0]}), model
        )
        variables["input"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )
        variables["target_type"] = ValueVariablesGroup(
            {
                "target_a": table["target_a"],
                "target_b": table["target_b"],
            }
        )

        translator = CastLikeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError,
            match="CastLike currently only supports casting to a single column type, not a group",
        ):
            translator.process()


class TestIdentityTranslator:

    def test_identity_single_column_passthrough(self):
        """Test IdentityTranslator passes a single column through unchanged."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Identity(input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = IdentityTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Verify output equals input
        backend = ibis.duckdb.connect()
        input_values = list(backend.execute(table["input"]))
        output_values = list(backend.execute(result))
        assert output_values == input_values

    def test_identity_group_columns_passthrough(self):
        """Test IdentityTranslator passes a NumericVariablesGroup through unchanged."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
                "col_c": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Identity(input)
            }
        """)

        # Use dummy table for GraphVariables since we override the input
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
                "col_c": table["col_c"],
            }
        )

        translator = IdentityTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Should return the same NumericVariablesGroup
        assert isinstance(result, NumericVariablesGroup)
        assert len(result) == 3
        assert "col_a" in result
        assert "col_b" in result
        assert "col_c" in result

        # Verify all columns preserved with same values
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["col_b"])) == [4.0, 5.0, 6.0]
        assert list(backend.execute(result["col_c"])) == [7.0, 8.0, 9.0]


class TestLinearClassifierTranslator:

    def test_linear_classifier_binary_classification(self):
        """Test LinearClassifierTranslator with binary classification."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [0.5, 0.5, -0.5, -0.5],
                    intercepts: floats = [1.0, -1.0],
                    classlabels_ints: ints = [0, 1]
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "feature1": table["feature1"],
                "feature2": table["feature2"],
            }
        )

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "Y" in variables
        assert "Z" in variables

        predictions = variables.peek_variable("Y")
        scores = variables.peek_variable("Z")

        assert isinstance(scores, ValueVariablesGroup)
        assert "0" in scores
        assert "1" in scores

        backend = ibis.duckdb.connect()
        # Class 0 has higher score so it should be predicted
        assert backend.execute(predictions)[0] == "0"

    def test_linear_classifier_multiclass(self):
        """Test LinearClassifierTranslator with multi-class classification (3+ classes)."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,3] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [1.0, 0.0, 0.0, 1.0, -1.0, -1.0],
                    intercepts: floats = [0.0, 0.5, -0.5],
                    classlabels_ints: ints = [0, 1, 2]
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "feature1": table["feature1"],
                "feature2": table["feature2"],
            }
        )

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        predictions = variables.peek_variable("Y")
        scores = variables.peek_variable("Z")

        assert isinstance(scores, ValueVariablesGroup)
        assert len(scores) == 3

        backend = ibis.duckdb.connect()
        # Class 1 has highest score for first row
        assert backend.execute(predictions)[0] == "1"

    def test_linear_classifier_with_intercepts(self):
        """Test LinearClassifierTranslator properly adds intercepts."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [1.0, -1.0],
                    intercepts: floats = [10.0, 20.0],
                    classlabels_ints: ints = [0, 1]
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        scores = variables.peek_variable("Z")

        backend = ibis.duckdb.connect()
        # Verify intercepts are properly added to scores
        assert backend.execute(scores["0"])[0] == 11.0
        assert backend.execute(scores["1"])[0] == 19.0

    def test_linear_classifier_post_transform_logistic(self):
        """Test LinearClassifierTranslator with post_transform='LOGISTIC'."""
        table = ibis.memtable({"input": [0.0, 1.0, -1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [1.0, -1.0],
                    intercepts: floats = [0.0, 0.0],
                    classlabels_ints: ints = [0, 1],
                    post_transform: string = "LOGISTIC"
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        scores = variables.peek_variable("Z")

        backend = ibis.duckdb.connect()
        # Logistic of 0.0 yields 0.5
        assert abs(backend.execute(scores["0"])[0] - 0.5) < 1e-6

    def test_linear_classifier_post_transform_softmax(self):
        """Test LinearClassifierTranslator with post_transform='SOFTMAX'."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [1.0, 0.0, 0.0, 1.0],
                    intercepts: floats = [0.0, 0.0],
                    classlabels_ints: ints = [0, 1],
                    post_transform: string = "SOFTMAX"
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "feature1": table["feature1"],
                "feature2": table["feature2"],
            }
        )

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        scores = variables.peek_variable("Z")
        assert isinstance(scores, ValueVariablesGroup)
        assert "0" in scores
        assert "1" in scores

    def test_linear_classifier_string_labels(self):
        """Test LinearClassifierTranslator with classlabels_strings instead of classlabels_ints."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (string[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [1.0, -1.0],
                    intercepts: floats = [0.0, 0.0],
                    classlabels_strings: strings = ["cat", "dog"]
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        predictions = variables.peek_variable("Y")
        scores = variables.peek_variable("Z")

        assert "cat" in scores
        assert "dog" in scores

        backend = ibis.duckdb.connect()
        # "cat" has higher score so it should be predicted
        assert backend.execute(predictions)[0] == "cat"

    def test_linear_classifier_single_input(self):
        """Test LinearClassifierTranslator when input is a single column (not a VariablesGroup)."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [2.0, -2.0],
                    intercepts: floats = [0.0, 0.0],
                    classlabels_ints: ints = [0, 1]
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        predictions = variables.peek_variable("Y")

        backend = ibis.duckdb.connect()
        # Class 0: 1.0*2.0 = 2.0
        # Class 1: 1.0*(-2.0) = -2.0
        # Prediction should be class 0
        assert backend.execute(predictions)[0] == "0"

    def test_linear_classifier_missing_classlabels(self):
        """Test LinearClassifierTranslator raises error when no classlabels defined."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [1.0, -1.0],
                    intercepts: floats = [0.0, 0.0]
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="LinearClassifier: classlabels_ints or classlabels_strings must be defined",
        ):
            translator.process()

    def test_linear_classifier_coefficient_mismatch(self):
        """Test LinearClassifierTranslator raises error when coefficients length doesn't match classes × features."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [1.0, 2.0, 3.0],
                    classlabels_ints: ints = [0, 1]
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "feature1": table["feature1"],
                "feature2": table["feature2"],
            }
        )

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="Coefficients length must equal number of classes × number of input fields",
        ):
            translator.process()

    def test_linear_classifier_multi_class_mode_not_implemented(self):
        """Test LinearClassifierTranslator raises error when multi_class != 0."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [1.0, -1.0],
                    classlabels_ints: ints = [0, 1],
                    multi_class: int = 1
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError, match="Multi-class classification is not implemented"
        ):
            translator.process()


class TestLinearRegressorTranslator:

    def test_linear_regressor_single_target(self):
        """Test LinearRegressorTranslator with single target regression (targets=1)."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [0.5, 0.5],
                    intercepts: floats = [10.0],
                    targets: int = 1
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "feature1": table["feature1"],
                "feature2": table["feature2"],
            }
        )

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "Y" in variables
        result = variables.peek_variable("Y")

        backend = ibis.duckdb.connect()
        assert backend.execute(result["target_0"])[0] == 12.5

    def test_linear_regressor_multi_target(self):
        """Test LinearRegressorTranslator with multi-target regression (targets=2+)."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N,2] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [1.0, 0.0, 0.0, 1.0],
                    intercepts: floats = [5.0, 10.0],
                    targets: int = 2
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "feature1": table["feature1"],
                "feature2": table["feature2"],
            }
        )

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("Y")
        assert isinstance(result, ValueVariablesGroup)
        assert "target_0" in result
        assert "target_1" in result

        backend = ibis.duckdb.connect()
        assert backend.execute(result["target_0"])[0] == 6.0
        assert backend.execute(result["target_1"])[0] == 14.0

    def test_linear_regressor_with_intercepts(self):
        """Test LinearRegressorTranslator properly adds intercepts."""
        table = ibis.memtable({"feature": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [2.0],
                    intercepts: floats = [100.0]
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup({"feature": table["feature"]})

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("Y")

        backend = ibis.duckdb.connect()
        # Verify intercept is properly added
        assert backend.execute(result["target_0"])[0] == 102.0

    def test_linear_regressor_no_intercepts(self):
        """Test LinearRegressorTranslator without intercepts."""
        table = ibis.memtable({"feature": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [3.0]
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup({"feature": table["feature"]})

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("Y")

        backend = ibis.duckdb.connect()
        # Default intercept is 0
        assert backend.execute(result["target_0"])[0] == 3.0

    def test_linear_regressor_single_column_input(self):
        """Test LinearRegressorTranslator with single column input (not VariablesGroup)."""
        table = ibis.memtable({"input": [2.0, 3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [5.0],
                    intercepts: floats = [1.0]
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("Y")

        backend = ibis.duckdb.connect()
        # Prediction: 2.0*5.0 + 1.0 = 11.0
        assert backend.execute(result)[0] == 11.0

    def test_linear_regressor_coefficient_mismatch(self):
        """Test LinearRegressorTranslator raises error when coefficients length mismatch."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [1.0, 2.0, 3.0],
                    targets: int = 1
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "feature1": table["feature1"],
                "feature2": table["feature2"],
            }
        )

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="Coefficients length must equal targets number of input fields",
        ):
            translator.process()

    def test_linear_regressor_intercepts_length_mismatch(self):
        """Test LinearRegressorTranslator raises error when intercepts length mismatch."""
        table = ibis.memtable({"feature": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N,2] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [1.0, 2.0],
                    intercepts: floats = [5.0],
                    targets: int = 2
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup({"feature": table["feature"]})

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="LinearRegressor: intercepts length must match targets or be empty",
        ):
            translator.process()

    def test_linear_regressor_post_transform_not_implemented(self):
        """Test LinearRegressorTranslator raises error when post_transform != 'NONE'."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [1.0],
                    post_transform: string = "LOGISTIC"
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError, match="Post transform is not implemented"
        ):
            translator.process()

    def test_linear_regressor_single_input_multiple_targets(self):
        """Test LinearRegressorTranslator raises error with single input and multiple targets."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N,2] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [1.0, 2.0],
                    intercepts: floats = [0.0, 0.0],
                    targets: int = 2
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="Single column input expects exactly one target and one coefficient",
        ):
            translator.process()




class TestTreeEnsembleClassifierTranslator:

    def test_binary_classification_single_tree(self):
        """Test TreeEnsembleClassifier with binary classification and single tree."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.classifier import TreeEnsembleClassifierTranslator

        table = ibis.memtable({"X": [0.3, 0.7, 0.2]})

        # Create a simple binary classification tree:
        # if feature[0] <= 0.5:
        #     return class 0 (weight 1.0)
        # else:
        #     return class 1 (weight 1.0)
        node = helper.make_node(
            op_type="TreeEnsembleClassifier",
            inputs=["X"],
            outputs=["Y", "P"],
            domain="ai.onnx.ml",
            # Tree structure attributes
            nodes_treeids=[0, 0, 0],
            nodes_nodeids=[0, 1, 2],
            nodes_featureids=[0, 0, 0],
            nodes_modes=["BRANCH_LEQ", "LEAF", "LEAF"],
            nodes_values=[0.5, 0.0, 0.0],
            nodes_truenodeids=[1, 0, 0],
            nodes_falsenodeids=[2, 0, 0],
            nodes_missing_value_tracks_true=[0, 0, 0],
            # Class weights
            class_treeids=[0, 0],
            class_nodeids=[1, 2],
            class_ids=[0, 0],
            class_weights=[1.0, -1.0],
            # Class labels
            classlabels_int64s=[0, 1],
            post_transform="NONE",
        )

        # Create a minimal graph and model
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [None])
        P = helper.make_tensor_value_info("P", TensorProto.FLOAT, [None, 2])

        graph = helper.make_graph([node], "test", [X], [Y, P])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)

        translator = TreeEnsembleClassifierTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "Y" in variables
        assert "P" in variables

        label_result = variables.peek_variable("Y")
        prob_result = variables.peek_variable("P")

        backend = ibis.duckdb.connect()
        labels = list(backend.execute(label_result))
        assert labels == [1, 0, 1]

        # Check probabilities
        assert isinstance(prob_result, NumericVariablesGroup)
        assert "0" in prob_result
        assert "1" in prob_result

    def test_multiclass_classification_single_tree(self):
        """Test TreeEnsembleClassifier with multi-class classification."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.classifier import TreeEnsembleClassifierTranslator

        table = ibis.memtable({"X": [0.3, 0.7, 0.2]})

        # Create a multi-class tree with 3 classes
        node = helper.make_node(
            op_type="TreeEnsembleClassifier",
            inputs=["X"],
            outputs=["Y", "P"],
            domain="ai.onnx.ml",
            # Simple tree: always returns class weights at leaf 1
            nodes_treeids=[0, 0],
            nodes_nodeids=[0, 1],
            nodes_featureids=[0, 0],
            nodes_modes=["BRANCH_LEQ", "LEAF"],
            nodes_values=[10.0, 0.0],
            nodes_truenodeids=[1, 0],
            nodes_falsenodeids=[1, 0],
            nodes_missing_value_tracks_true=[0, 0],
            # Class weights for 3 classes at leaf node
            class_treeids=[0, 0, 0],
            class_nodeids=[1, 1, 1],
            class_ids=[0, 1, 2],
            class_weights=[0.2, 0.5, 0.3],
            # String class labels
            classlabels_strings=["cat", "dog", "bird"],
            post_transform="NONE",
        )

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.STRING, [None])
        P = helper.make_tensor_value_info("P", TensorProto.FLOAT, [None, 3])

        graph = helper.make_graph([node], "test", [X], [Y, P])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)

        translator = TreeEnsembleClassifierTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "Y" in variables
        assert "P" in variables

        label_result = variables.peek_variable("Y")
        prob_result = variables.peek_variable("P")

        backend = ibis.duckdb.connect()
        labels = list(backend.execute(label_result))
        # All inputs should predict "dog" since it has highest weight (0.5)
        assert labels == ["dog", "dog", "dog"]

        # Check probabilities group structure
        assert isinstance(prob_result, NumericVariablesGroup)
        assert "cat" in prob_result
        assert "dog" in prob_result
        assert "bird" in prob_result

    def test_classifier_invalid_input_type(self):
        """Test TreeEnsembleClassifier raises error for invalid input type."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.classifier import TreeEnsembleClassifierTranslator

        table = ibis.memtable({"X": [0.3, 0.7, 0.2]})

        node = helper.make_node(
            op_type="TreeEnsembleClassifier",
            inputs=["X"],
            outputs=["Y", "P"],
            domain="ai.onnx.ml",
            nodes_treeids=[0],
            nodes_nodeids=[0],
            nodes_featureids=[0],
            nodes_modes=["LEAF"],
            nodes_values=[0.0],
            nodes_truenodeids=[0],
            nodes_falsenodeids=[0],
            nodes_missing_value_tracks_true=[0],
            class_treeids=[0],
            class_nodeids=[0],
            class_ids=[0],
            class_weights=[1.0],
            classlabels_int64s=[0, 1],
        )

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [None])
        P = helper.make_tensor_value_info("P", TensorProto.FLOAT, [None, 2])

        graph = helper.make_graph([node], "test", [X], [Y, P])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)
        variables["X"] = "invalid_string_input"  # type: ignore[assignment]

        translator = TreeEnsembleClassifierTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="TreeEnsembleClassifier: The first operand must be a column or a column group"
        ):
            translator.process()


class TestTreeEnsembleRegressorTranslator:

    def test_single_tree_regression(self):
        """Test TreeEnsembleRegressor with a single decision tree."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.regressor import TreeEnsembleRegressorTranslator

        table = ibis.memtable({"X": [0.3, 0.7, 0.2]})

        # Create a simple regression tree:
        # if feature[0] <= 0.5:
        #     return 10.0
        # else:
        #     return 20.0
        node = helper.make_node(
            op_type="TreeEnsembleRegressor",
            inputs=["X"],
            outputs=["Y"],
            domain="ai.onnx.ml",
            # Tree structure
            nodes_treeids=[0, 0, 0],
            nodes_nodeids=[0, 1, 2],
            nodes_featureids=[0, 0, 0],
            nodes_modes=["BRANCH_LEQ", "LEAF", "LEAF"],
            nodes_values=[0.5, 0.0, 0.0],
            nodes_truenodeids=[1, 0, 0],
            nodes_falsenodeids=[2, 0, 0],
            nodes_missing_value_tracks_true=[0, 0, 0],
            # Target weights for regression
            target_treeids=[0, 0],
            target_nodeids=[1, 2],
            target_weights=[10.0, 20.0],
            # Base value (offset)
            base_values=[5.0],
        )

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)

        translator = TreeEnsembleRegressorTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "Y" in variables
        result = variables.peek_variable("Y")

        backend = ibis.duckdb.connect()
        predictions = list(backend.execute(result))
        assert predictions == [15.0, 25.0, 15.0]

    def test_regression_base_values_applied(self):
        """Test TreeEnsembleRegressor correctly applies base_values."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.regressor import TreeEnsembleRegressorTranslator

        table = ibis.memtable({"X": [1.0, 2.0, 3.0]})

        # Simple tree that always returns same value
        node = helper.make_node(
            op_type="TreeEnsembleRegressor",
            inputs=["X"],
            outputs=["Y"],
            domain="ai.onnx.ml",
            # Single leaf node tree
            nodes_treeids=[0],
            nodes_nodeids=[0],
            nodes_featureids=[0],
            nodes_modes=["LEAF"],
            nodes_values=[0.0],
            nodes_truenodeids=[0],
            nodes_falsenodeids=[0],
            nodes_missing_value_tracks_true=[0],
            # Weight at leaf
            target_treeids=[0],
            target_nodeids=[0],
            target_weights=[7.0],
            # Base value should be added
            base_values=[3.0],
        )

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)

        translator = TreeEnsembleRegressorTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "Y" in variables
        result = variables.peek_variable("Y")

        backend = ibis.duckdb.connect()
        computed = backend.execute(result)
        # Single-leaf tree returns a constant expression (scalar)
        assert computed == 10.0

    def test_regressor_invalid_input_type(self):
        """Test TreeEnsembleRegressor raises error for invalid input type."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.regressor import TreeEnsembleRegressorTranslator

        table = ibis.memtable({"X": [1.0, 2.0, 3.0]})

        node = helper.make_node(
            op_type="TreeEnsembleRegressor",
            inputs=["X"],
            outputs=["Y"],
            domain="ai.onnx.ml",
            nodes_treeids=[0],
            nodes_nodeids=[0],
            nodes_featureids=[0],
            nodes_modes=["LEAF"],
            nodes_values=[0.0],
            nodes_truenodeids=[0],
            nodes_falsenodeids=[0],
            nodes_missing_value_tracks_true=[0],
            target_treeids=[0],
            target_nodeids=[0],
            target_weights=[1.0],
        )

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)
        variables["X"] = "invalid_string_input"  # type: ignore[assignment]

        translator = TreeEnsembleRegressorTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="TreeEnsembleRegressor: The first operand must be a column or a column group"
        ):
            translator.process()




class TestScalerTranslator:

    def test_scaler_single_column(self):
        """Test ScalerTranslator with single column scaling."""
        table = ibis.memtable({"input": [2.0, 4.0, 6.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Scaler <offset: floats = [1.0], scale: floats = [2.0]> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ScalerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Y = (X - offset) * scale = (X - 1) * 2
        # [2, 4, 6] -> [2, 6, 10]
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result)) == [2.0, 6.0, 10.0]

    def test_scaler_group_columns(self):
        """Test ScalerTranslator with group of columns."""
        table = ibis.memtable(
            {
                "col_a": [2.0, 4.0, 6.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Scaler <offset: floats = [1.0, 5.0], scale: floats = [2.0, 0.5]> (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = ScalerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        # col_a: Y = (X - 1) * 2 = [2, 6, 10]
        assert list(backend.execute(result["col_a"])) == [2.0, 6.0, 10.0]
        # col_b: Y = (X - 5) * 0.5 = [2.5, 7.5, 12.5]
        assert list(backend.execute(result["col_b"])) == [2.5, 7.5, 12.5]

    def test_scaler_only_offset(self):
        """Test ScalerTranslator with only offset (scale=1.0)."""
        table = ibis.memtable({"input": [2.0, 4.0, 6.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Scaler <offset: floats = [3.0], scale: floats = [1.0]> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ScalerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Y = (X - 3) * 1 = X - 3
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result)) == [-1.0, 1.0, 3.0]

    def test_scaler_only_scale(self):
        """Test ScalerTranslator with only scale (offset=0.0)."""
        table = ibis.memtable({"input": [2.0, 4.0, 6.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Scaler <offset: floats = [0.0], scale: floats = [3.0]> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ScalerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Y = (X - 0) * 3 = X * 3
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result)) == [6.0, 12.0, 18.0]

    def test_scaler_mismatched_offset_scale_counts(self):
        """Test ScalerTranslator raises error when offset/scale counts don't match."""
        table = ibis.memtable(
            {
                "col_a": [2.0, 4.0, 6.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Scaler <offset: floats = [1.0, 2.0, 3.0], scale: floats = [2.0, 3.0]> (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = ScalerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="offset and scale lists must match"):
            translator.process()


class TestOneHotEncoderTranslator:

    def test_onehot_string_column(self):
        """Test OneHotEncoderTranslator with string column."""
        table = ibis.memtable({"category": ["cat", "dog", "cat", "bird"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] category) => (float[N, 3] output) {
                output = ai.onnx.ml.OneHotEncoder <cats_strings: strings = ["cat", "dog", "bird"]> (category)
            }
        """)

        variables = GraphVariables(table, model)
        translator = OneHotEncoderTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        # First row is "cat", so cat=1.0, dog=0.0, bird=0.0
        assert list(backend.execute(result["cat"])) == [1.0, 0.0, 1.0, 0.0]
        assert list(backend.execute(result["dog"])) == [0.0, 1.0, 0.0, 0.0]
        assert list(backend.execute(result["bird"])) == [0.0, 0.0, 0.0, 1.0]

    def test_onehot_output_type(self):
        """Test OneHotEncoderTranslator output is ValueVariablesGroup with correct keys."""
        table = ibis.memtable({"category": ["apple", "banana", "apple"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] category) => (float[N, 2] output) {
                output = ai.onnx.ml.OneHotEncoder <cats_strings: strings = ["apple", "banana"]> (category)
            }
        """)

        variables = GraphVariables(table, model)
        translator = OneHotEncoderTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert "apple" in result
        assert "banana" in result
        assert len(result) == 2

    def test_onehot_values(self):
        """Test OneHotEncoderTranslator outputs 0.0 or 1.0 floats."""
        table = ibis.memtable({"category": ["red", "blue", "green", "red"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] category) => (float[N, 3] output) {
                output = ai.onnx.ml.OneHotEncoder <cats_strings: strings = ["red", "green", "blue"]> (category)
            }
        """)

        variables = GraphVariables(table, model)
        translator = OneHotEncoderTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        # Input: ["red", "blue", "green", "red"]
        # Verify one-hot encoding produces correct 0.0/1.0 values
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["red"])) == [1.0, 0.0, 0.0, 1.0]
        assert list(backend.execute(result["green"])) == [0.0, 0.0, 1.0, 0.0]
        assert list(backend.execute(result["blue"])) == [0.0, 1.0, 0.0, 0.0]

    def test_onehot_missing_cats_strings(self):
        """Test OneHotEncoderTranslator raises error when cats_strings is missing."""
        table = ibis.memtable({"category": ["cat", "dog"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] category) => (float[N, 3] output) {
                output = ai.onnx.ml.OneHotEncoder (category)
            }
        """)

        variables = GraphVariables(table, model)
        translator = OneHotEncoderTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="attribute cats_strings not found"):
            translator.process()


class TestLabelEncoderTranslator:

    def test_labelencoder_string_to_int(self):
        """Test LabelEncoderTranslator encoding string labels to integers."""
        table = ibis.memtable({"label": ["cat", "dog", "cat", "bird", "dog"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] label) => (int64[N] output) {
                output = ai.onnx.ml.LabelEncoder <keys_strings: strings = ["cat", "dog", "bird"], values_int64s: ints = [0, 1, 2]> (label)
            }
        """)

        variables = GraphVariables(table, model)
        translator = LabelEncoderTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result)) == [0, 1, 0, 2, 1]

    def test_labelencoder_int_to_int(self):
        """Test LabelEncoderTranslator encoding integer labels to different integers."""
        table = ibis.memtable({"label": [10, 20, 10, 30, 20]})
        model = onnx.parser.parse_graph("""
            agraph (int64[N] label) => (int64[N] output) {
                output = ai.onnx.ml.LabelEncoder <keys_int64s: ints = [10, 20, 30], values_int64s: ints = [100, 200, 300]> (label)
            }
        """)

        variables = GraphVariables(table, model)
        translator = LabelEncoderTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result)) == [100, 200, 100, 300, 200]

    def test_labelencoder_default_value(self):
        """Test LabelEncoderTranslator handles default value for unknown labels."""
        table = ibis.memtable({"label": ["cat", "dog", "unknown", "cat"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] label) => (int64[N] output) {
                output = ai.onnx.ml.LabelEncoder <keys_strings: strings = ["cat", "dog"], values_int64s: ints = [0, 1], default_int64: int = 999> (label)
            }
        """)

        variables = GraphVariables(table, model)
        translator = LabelEncoderTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result)) == [0, 1, 999, 0]

    def test_labelencoder_missing_keys(self):
        """Test LabelEncoderTranslator raises error when keys attribute is missing."""
        table = ibis.memtable({"label": ["cat", "dog"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] label) => (int64[N] output) {
                output = ai.onnx.ml.LabelEncoder <values_int64s: ints = [0, 1]> (label)
            }
        """)

        variables = GraphVariables(table, model)
        translator = LabelEncoderTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="required mapping attributes not found"):
            translator.process()

class TestZipMapTranslator:

    def test_zipmap_string_labels_group(self):
        """Test ZipMapTranslator with string labels and group of columns."""
        table = ibis.memtable(
            {
                "prob_class_0": [0.7, 0.2, 0.4],
                "prob_class_1": [0.2, 0.5, 0.1],
                "prob_class_2": [0.1, 0.3, 0.5],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output) {
                output = ai.onnx.ml.ZipMap <classlabels_strings: strings = ["negative", "neutral", "positive"]> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = ValueVariablesGroup(
            {
                "class_0": table["prob_class_0"],
                "class_1": table["prob_class_1"],
                "class_2": table["prob_class_2"],
            }
        )

        translator = ZipMapTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        # Check that labels were properly mapped
        assert "negative" in result
        assert "neutral" in result
        assert "positive" in result

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["negative"])) == [0.7, 0.2, 0.4]
        assert list(backend.execute(result["neutral"])) == [0.2, 0.5, 0.1]
        assert list(backend.execute(result["positive"])) == [0.1, 0.3, 0.5]

    def test_zipmap_int64_labels_group(self):
        """Test ZipMapTranslator with int64 labels and group of columns."""
        table = ibis.memtable(
            {
                "feat_0": [1.0, 2.0, 3.0],
                "feat_1": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output) {
                output = ai.onnx.ml.ZipMap <classlabels_int64s: ints = [100, 200]> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = ValueVariablesGroup(
            {
                "feat_0": table["feat_0"],
                "feat_1": table["feat_1"],
            }
        )

        translator = ZipMapTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        # int64 labels are converted to strings
        assert "100" in result
        assert "200" in result

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["100"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["200"])) == [4.0, 5.0, 6.0]

    def test_zipmap_single_column(self):
        """Test ZipMapTranslator with single column and single label."""
        table = ibis.memtable({"prob": [0.95, 0.87, 0.92]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output) {
                output = ai.onnx.ml.ZipMap <classlabels_strings: strings = ["confidence"]> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = table["prob"]

        translator = ZipMapTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        assert "confidence" in result
        assert len(result) == 1

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["confidence"])) == [0.95, 0.87, 0.92]

    def test_zipmap_missing_labels_error(self):
        """Test ZipMapTranslator raises error when no classlabels attribute is found."""
        table = ibis.memtable({"data": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output) {
                output = ai.onnx.ml.ZipMap (data)
            }
        """)

        variables = GraphVariables(table, model)

        translator = ZipMapTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError, match="ZipMap: required mapping attributes not found"
        ):
            translator.process()

    def test_zipmap_mismatched_label_count(self):
        """Test ZipMapTranslator raises error when labels count doesn't match columns."""
        table = ibis.memtable(
            {
                "col1": [1.0, 2.0, 3.0],
                "col2": [4.0, 5.0, 6.0],
                "col3": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output) {
                output = ai.onnx.ml.ZipMap <classlabels_strings: strings = ["label1", "label2"]> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = ValueVariablesGroup(
            {
                "col1": table["col1"],
                "col2": table["col2"],
                "col3": table["col3"],
            }
        )

        translator = ZipMapTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError, match="ZipMap: The number of labels and columns must match"
        ):
            translator.process()

