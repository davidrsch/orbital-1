"""Tests for Cast CastLike Identity pipeline step translators."""

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
    """Create an ONNX GraphProto with given node, I/O specs, and initializers."""
    return helper.make_graph(
        [node],
        "test_graph",
        inputs_info,
        outputs_info,
        initializer=initializers,
    )

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



