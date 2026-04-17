"""Tests for FeatureVectorizer and Concat pipeline step translators."""

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
from orbital.translation.steps.concat import ConcatTranslator
from orbital.translation.steps.featurevectorizer import FeatureVectorizerTranslator


from conftest import make_graph_with_inits as _make_graph_with_inits


class TestFeatureVectorizerTranslator:
    def test_featurevectorizer_multiple_inputs_with_dimensions(self):
        """Test vectorizing multiple inputs with inputdimensions attribute."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "group_col_a": [4.0, 5.0, 6.0],
                "group_col_b": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input1, float[N] input2) => (float[N] output) {
                output = ai.onnx.ml.FeatureVectorizer <inputdimensions: ints = [1, 2]> (input1, input2)
            }
        """)

        variables = GraphVariables(
            ibis.memtable({"input1": [1.0], "input2": [1.0]}), model
        )
        variables["input1"] = table["feature1"]
        variables["input2"] = ValueVariablesGroup(
            {
                "col_a": table["group_col_a"],
                "col_b": table["group_col_b"],
            }
        )

        translator = FeatureVectorizerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 3
        assert "input1" in result
        assert "input2.col_a" in result
        assert "input2.col_b" in result

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["input1"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["input2.col_a"])) == [4.0, 5.0, 6.0]
        assert list(backend.execute(result["input2.col_b"])) == [7.0, 8.0, 9.0]

    def test_featurevectorizer_single_input_feature(self):
        """Test single input feature."""
        table = ibis.memtable({"input1": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input1) => (float[N] output) {
                output = ai.onnx.ml.FeatureVectorizer <inputdimensions: ints = [1]> (input1)
            }
        """)

        variables = GraphVariables(table, model)
        translator = FeatureVectorizerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 1
        assert "input1" in result

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["input1"])) == [1.0, 2.0, 3.0]

    def test_featurevectorizer_mismatched_dimensions(self):
        """Test error when input dimensions don't match actual columns."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "group_col_a": [4.0, 5.0, 6.0],
                "group_col_b": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input1, float[N] input2) => (float[N] output) {
                output = ai.onnx.ml.FeatureVectorizer <inputdimensions: ints = [1, 3]> (input1, input2)
            }
        """)

        variables = GraphVariables(
            ibis.memtable({"input1": [1.0], "input2": [1.0]}), model
        )
        variables["input1"] = table["feature1"]
        variables["input2"] = ValueVariablesGroup(
            {
                "col_a": table["group_col_a"],
                "col_b": table["group_col_b"],
            }
        )

        translator = FeatureVectorizerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="Number of columns in input input2"):
            translator.process()

    def test_featurevectorizer_wrong_single_column_dimension(self):
        """Test error when single column has wrong dimension."""
        table = ibis.memtable({"input1": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input1) => (float[N] output) {
                output = ai.onnx.ml.FeatureVectorizer <inputdimensions: ints = [2]> (input1)
            }
        """)

        variables = GraphVariables(table, model)
        translator = FeatureVectorizerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="When merging over individual columns, the dimension should be 1",
        ):
            translator.process()

    def test_featurevectorizer_mismatched_input_count(self):
        """Test error when input count doesn't match dimensions count."""
        table = ibis.memtable({"input1": [1.0, 2.0, 3.0], "input2": [4.0, 5.0, 6.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input1, float[N] input2) => (float[N] output) {
                output = ai.onnx.ml.FeatureVectorizer <inputdimensions: ints = [1]> (input1, input2)
            }
        """)

        variables = GraphVariables(table, model)
        translator = FeatureVectorizerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="Number of input dimensions should be equal to number of inputs",
        ):
            translator.process()


class TestConcatTranslator:
    def test_concat_two_single_columns(self):
        """Test concatenating two single columns."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] feature1, float[N] feature2) => (float[N] output) {
                output = Concat <axis: int = 1> (feature1, feature2)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ConcatTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 2
        assert "feature1" in result
        assert "feature2" in result

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["feature1"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["feature2"])) == [4.0, 5.0, 6.0]

    def test_concat_column_groups(self):
        """Test concatenating column groups."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
                "col_c": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] group1, float[N] group2) => (float[N] output) {
                output = Concat <axis: int = -1> (group1, group2)
            }
        """)

        variables = GraphVariables(
            ibis.memtable({"group1": [1.0], "group2": [1.0]}), model
        )
        variables["group1"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )
        variables["group2"] = ValueVariablesGroup(
            {
                "col_c": table["col_c"],
            }
        )

        translator = ConcatTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 3
        assert "group1.col_a" in result
        assert "group1.col_b" in result
        assert "group2.col_c" in result

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["group1.col_a"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["group1.col_b"])) == [4.0, 5.0, 6.0]
        assert list(backend.execute(result["group2.col_c"])) == [7.0, 8.0, 9.0]

    def test_concat_mix_single_columns_and_groups(self):
        """Test concatenating mix of single columns and groups."""
        table = ibis.memtable(
            {
                "single_col": [1.0, 2.0, 3.0],
                "group_col_a": [4.0, 5.0, 6.0],
                "group_col_b": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] single, float[N] group) => (float[N] output) {
                output = Concat <axis: int = 1> (single, group)
            }
        """)

        variables = GraphVariables(
            ibis.memtable({"single": [1.0], "group": [1.0]}), model
        )
        variables["single"] = table["single_col"]
        variables["group"] = ValueVariablesGroup(
            {
                "col_a": table["group_col_a"],
                "col_b": table["group_col_b"],
            }
        )

        translator = ConcatTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 3
        assert "single" in result
        assert "group.col_a" in result
        assert "group.col_b" in result

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["single"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["group.col_a"])) == [4.0, 5.0, 6.0]
        assert list(backend.execute(result["group.col_b"])) == [7.0, 8.0, 9.0]

    def test_concat_unsupported_axis(self):
        """Test error for unsupported axis."""
        table = ibis.memtable(
            {"feature1": [1.0, 2.0, 3.0], "feature2": [4.0, 5.0, 6.0]}
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] feature1, float[N] feature2) => (float[N] output) {
                output = Concat <axis: int = 0> (feature1, feature2)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ConcatTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError, match="only supports concatenating over columns"
        ):
            translator.process()
