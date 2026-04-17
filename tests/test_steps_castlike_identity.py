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
from orbital.translation.steps.identity import IdentityTranslator
from orbital.translation.steps.cast import CastLikeTranslator


from conftest import make_graph_with_inits as _make_graph_with_inits


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
