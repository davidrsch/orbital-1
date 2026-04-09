"""Tests for ML normalization and encoding op pipeline step translators."""

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

