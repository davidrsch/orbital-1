"""Tests for LinearClassifier pipeline step translators."""

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
