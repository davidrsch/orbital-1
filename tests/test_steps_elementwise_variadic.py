"""Tests for variadic/reduction arithmetic pipeline step translators (Sum, Max, Min, Mean)."""

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


from conftest import make_graph_with_inits as _make_graph_with_inits

# ---------------------------------------------------------------------------
# Variadic element-wise operators: Sum, Max, Min, Mean
# ---------------------------------------------------------------------------


def _make_variadic_node(op: str, n_inputs: int, output_name: str = "output"):
    """Build an ONNX NodeProto for a variadic op with n_inputs."""
    from onnx import helper

    input_names = [f"inp{i}" for i in range(n_inputs)]
    return helper.make_node(op, input_names, [output_name])


class TestSumTranslator:
    """Tests for VariadicSumTranslator (ONNX 'Sum' op)."""

    def test_variadic_sum_registered(self):
        from orbital.translation.steps.variadicsum import VariadicSumTranslator

        assert TRANSLATORS.get("Sum") is VariadicSumTranslator

    def test_variadic_sum_two_columns(self):
        """Sum of two scalar inputs returns element-wise addition."""
        table = ibis.memtable({"inp0": [1.0, 2.0], "inp1": [3.0, 4.0]})
        from onnx import helper

        node = helper.make_node("Sum", ["inp0", "inp1"], ["output"])
        graph = helper.make_graph(
            [node],
            "g",
            [
                helper.make_tensor_value_info("inp0", 1, [None]),
                helper.make_tensor_value_info("inp1", 1, [None]),
            ],
            [helper.make_tensor_value_info("output", 1, [None])],
        )
        variables = GraphVariables(table, graph)
        from orbital.translation.steps.variadicsum import VariadicSumTranslator

        t = VariadicSumTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [4.0, 6.0]

    def test_variadic_sum_three_groups(self):
        """Sum of three VariablesGroups returns per-column sums."""
        table = ibis.memtable({"a": [1.0], "b": [2.0], "c": [3.0]})
        from onnx import helper

        node = helper.make_node("Sum", ["g0", "g1", "g2"], ["output"])
        graph = helper.make_graph(
            [node],
            "g",
            [],
            [helper.make_tensor_value_info("output", 1, [None])],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["g0"] = NumericVariablesGroup({"col0": table["a"]})
        variables["g1"] = NumericVariablesGroup({"col0": table["b"]})
        variables["g2"] = NumericVariablesGroup({"col0": table["c"]})
        from orbital.translation.steps.variadicsum import VariadicSumTranslator

        t = VariadicSumTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert list(backend.execute(list(result.values())[0])) == [6.0]


class TestMaxTranslator:
    """Tests for VariadicMaxTranslator (ONNX 'Max' op)."""

    def test_variadic_max_registered(self):
        from orbital.translation.steps.variadicmax import VariadicMaxTranslator

        assert TRANSLATORS.get("Max") is VariadicMaxTranslator

    def test_variadic_max_two_scalars(self):
        """Max of two columns returns per-row maximum."""
        table = ibis.memtable({"inp0": [1.0, 5.0], "inp1": [3.0, 2.0]})
        from onnx import helper

        node = helper.make_node("Max", ["inp0", "inp1"], ["output"])
        graph = helper.make_graph(
            [node],
            "g",
            [
                helper.make_tensor_value_info("inp0", 1, [None]),
                helper.make_tensor_value_info("inp1", 1, [None]),
            ],
            [helper.make_tensor_value_info("output", 1, [None])],
        )
        variables = GraphVariables(table, graph)
        from orbital.translation.steps.variadicmax import VariadicMaxTranslator

        t = VariadicMaxTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [3.0, 5.0]


class TestMinTranslator:
    """Tests for VariadicMinTranslator (ONNX 'Min' op)."""

    def test_variadic_min_registered(self):
        from orbital.translation.steps.variadicmin import VariadicMinTranslator

        assert TRANSLATORS.get("Min") is VariadicMinTranslator

    def test_variadic_min_two_scalars(self):
        """Min of two columns returns per-row minimum."""
        table = ibis.memtable({"inp0": [1.0, 5.0], "inp1": [3.0, 2.0]})
        from onnx import helper

        node = helper.make_node("Min", ["inp0", "inp1"], ["output"])
        graph = helper.make_graph(
            [node],
            "g",
            [
                helper.make_tensor_value_info("inp0", 1, [None]),
                helper.make_tensor_value_info("inp1", 1, [None]),
            ],
            [helper.make_tensor_value_info("output", 1, [None])],
        )
        variables = GraphVariables(table, graph)
        from orbital.translation.steps.variadicmin import VariadicMinTranslator

        t = VariadicMinTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [1.0, 2.0]


class TestMeanTranslator:
    """Tests for VariadicMeanTranslator (ONNX 'Mean' op)."""

    def test_variadic_mean_registered(self):
        from orbital.translation.steps.variadicmean import VariadicMeanTranslator

        assert TRANSLATORS.get("Mean") is VariadicMeanTranslator

    def test_variadic_mean_three_scalars(self):
        """Mean of three columns returns per-row average."""
        table = ibis.memtable(
            {"inp0": [1.0, 2.0], "inp1": [3.0, 4.0], "inp2": [5.0, 6.0]}
        )
        from onnx import helper

        node = helper.make_node("Mean", ["inp0", "inp1", "inp2"], ["output"])
        graph = helper.make_graph(
            [node],
            "g",
            [
                helper.make_tensor_value_info("inp0", 1, [None]),
                helper.make_tensor_value_info("inp1", 1, [None]),
                helper.make_tensor_value_info("inp2", 1, [None]),
            ],
            [helper.make_tensor_value_info("output", 1, [None])],
        )
        variables = GraphVariables(table, graph)
        from orbital.translation.steps.variadicmean import VariadicMeanTranslator

        t = VariadicMeanTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        # Row 0: (1+3+5)/3 = 3.0; Row 1: (2+4+6)/3 = 4.0
        assert abs(result[0] - 3.0) < 1e-6
        assert abs(result[1] - 4.0) < 1e-6

    def test_variadic_mean_mismatched_groups_raises(self):
        """Mean of groups with different column counts should raise ValueError."""
        table = ibis.memtable({"a": [1.0], "b": [2.0], "c": [3.0]})
        from onnx import helper

        node = helper.make_node("Mean", ["g0", "g1"], ["output"])
        graph = helper.make_graph(
            [node],
            "g",
            [],
            [helper.make_tensor_value_info("output", 1, [None])],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["g0"] = NumericVariablesGroup(
            {"col0": table["a"], "col1": table["b"]}
        )
        variables["g1"] = NumericVariablesGroup({"col0": table["c"]})
        from orbital.translation.steps.variadicmean import VariadicMeanTranslator

        t = VariadicMeanTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        with pytest.raises(ValueError, match="same shape"):
            t.process()


# Note: Class names already match the ONNX op key form expected by
# TestStepCoverage (`Test{op_key}Translator`), so no aliasing is required.

