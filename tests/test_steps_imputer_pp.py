"""Tests for Imputer pipeline step translators."""

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
from orbital.translation.steps.imputer import ImputerTranslator


from conftest import make_graph_with_inits as _make_graph_with_inits


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
