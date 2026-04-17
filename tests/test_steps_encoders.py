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
from orbital.translation.steps.onehotencoder import OneHotEncoderTranslator
from orbital.translation.steps.labelencoder import LabelEncoderTranslator


from conftest import make_graph_with_inits as _make_graph_with_inits


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
