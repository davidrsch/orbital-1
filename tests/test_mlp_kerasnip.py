"""Tests for keras-nip MLP activation/norm pipeline step translators."""

import math

import ibis
import numpy as np
import onnx
import pandas as pd
import pytest
from onnx import TensorProto, helper
from orbital_testing_helpers import execute_sql
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import orbital
from orbital import types
from orbital.ast import parse_pipeline
from orbital.translation.optimizer import Optimizer
from orbital.translation.options import TranslationOptions
from orbital.translation.steps.exp import ExpTranslator
from orbital.translation.steps.gemm import GemmTranslator
from orbital.translation.steps.relu import ReluTranslator
from orbital.translation.steps.rmsnorm import RMSNormalizationTranslator
from orbital.translation.steps.sigmoid import SigmoidTranslator
from orbital.translation.steps.swish import SwishTranslator
from orbital.translation.steps.tanh import TanhTranslator
from orbital.translation.steps.thresholdedrelu import ThresholdedReluTranslator
from orbital.translation.variables import (
    GraphVariables,
    NumericVariablesGroup,
    ValueVariablesGroup,
)

# ---------------------------------------------------------------------------
# Unit tests: ReluTranslator
# ---------------------------------------------------------------------------



class TestExpTranslator:
    optimizer = Optimizer(enabled=False)

    def test_exp_single(self):
        """Exp(0) == 1.0."""
        table = ibis.memtable({"input": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Exp(input)
            }
        """)
        variables = GraphVariables(table, model)
        translator = ExpTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        values = backend.execute(result).tolist()
        assert values == pytest.approx([math.exp(0.0)])

    def test_exp_values(self):
        """Exp should return e^x for various inputs."""
        xs = [-1.0, 0.0, 1.0, 2.0]
        table = ibis.memtable({"input": xs})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Exp(input)
            }
        """)
        variables = GraphVariables(table, model)
        translator = ExpTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        values = backend.execute(result).tolist()
        assert values == pytest.approx([math.exp(x) for x in xs], rel=1e-6)


# ---------------------------------------------------------------------------
# Unit tests: ThresholdedReluTranslator
# ---------------------------------------------------------------------------


class TestThresholdedReluTranslator:
    optimizer = Optimizer(enabled=False)

    def test_threlu_custom_alpha(self):
        """ThresholdedRelu with alpha=2.0: x if x>2 else 0."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        node = helper.make_node("ThresholdedRelu", ["input"], ["output"], alpha=2.0)
        graph = helper.make_graph([node], "g", [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [None]),
        ], [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [None]),
        ])
        variables = GraphVariables(table, graph)
        translator = ThresholdedReluTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        values = backend.execute(result).tolist()
        assert values == pytest.approx([0.0, 0.0, 3.0])

    def test_threlu_default_alpha(self):
        """ThresholdedRelu with default alpha=1.0: passthrough for x > 1, else 0."""
        table = ibis.memtable({"input": [0.5, 1.0, 1.5]})
        node = helper.make_node("ThresholdedRelu", ["input"], ["output"])
        graph = helper.make_graph([node], "g", [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [None]),
        ], [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [None]),
        ])
        variables = GraphVariables(table, graph)
        translator = ThresholdedReluTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        values = backend.execute(result).tolist()
        # x=0.5 (≤1.0 → 0), x=1.0 (not > 1.0 → 0), x=1.5 (> 1.0 → 1.5)
        assert values == pytest.approx([0.0, 0.0, 1.5])


# ---------------------------------------------------------------------------
# Unit tests: SwishTranslator
# ---------------------------------------------------------------------------


class TestSwishTranslator:
    optimizer = Optimizer(enabled=False)

    def test_swish_zero(self):
        """Swish(0) == 0.0."""
        table = ibis.memtable({"input": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Swish(input)
            }
        """)
        variables = GraphVariables(table, model)
        translator = SwishTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        values = backend.execute(result).tolist()
        assert values == pytest.approx([0.0])

    def test_swish_values(self):
        """Swish(x) = x * sigmoid(x)."""
        xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
        table = ibis.memtable({"input": xs})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Swish(input)
            }
        """)
        variables = GraphVariables(table, model)
        translator = SwishTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        values = backend.execute(result).tolist()
        expected = [x / (1.0 + math.exp(-x)) for x in xs]
        assert values == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Unit tests: RMSNormalizationTranslator
# ---------------------------------------------------------------------------


class TestRMSNormalizationTranslator:
    optimizer = Optimizer(enabled=False)

    def test_rmsnorm_unit_scale(self):
        """RMSNorm with unit scale: y_c = x_c / rms(x)."""
        # rms([3, 4]) = sqrt((9+16)/2) = sqrt(12.5)
        table = ibis.memtable({"input": [0.0], "h0": [3.0], "h1": [4.0]})
        scale = np.array([1.0, 1.0], dtype=np.float32)
        scale_tensor = helper.make_tensor("scale", TensorProto.FLOAT, [2], scale)
        node = helper.make_node("RMSNormalization", ["input", "scale"], ["output"],
                                epsilon=0.0)
        graph = helper.make_graph([node], "rmsnorm_test",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
            initializer=[scale_tensor],
        )
        variables = GraphVariables(table, graph)
        variables["input"] = NumericVariablesGroup({
            "h0": table["h0"],
            "h1": table["h1"],
        })
        translator = RMSNormalizationTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        rms = math.sqrt((3.0**2 + 4.0**2) / 2.0)
        h0_val = backend.execute(result["h0"]).tolist()[0]
        h1_val = backend.execute(result["h1"]).tolist()[0]
        assert h0_val == pytest.approx(3.0 / rms, rel=1e-6)
        assert h1_val == pytest.approx(4.0 / rms, rel=1e-6)

    def test_rmsnorm_with_scale(self):
        """RMSNorm scales output by gamma per feature."""
        # rms([1, 1]) = 1.0; after scale=[2,3]: y = [2.0, 3.0]
        table = ibis.memtable({"input": [0.0], "h0": [1.0], "h1": [1.0]})
        scale = np.array([2.0, 3.0], dtype=np.float32)
        scale_tensor = helper.make_tensor("scale", TensorProto.FLOAT, [2], scale)
        node = helper.make_node("RMSNormalization", ["input", "scale"], ["output"],
                                epsilon=0.0)
        graph = helper.make_graph([node], "rmsnorm_scale_test",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
            initializer=[scale_tensor],
        )
        variables = GraphVariables(table, graph)
        variables["input"] = NumericVariablesGroup({
            "h0": table["h0"],
            "h1": table["h1"],
        })
        translator = RMSNormalizationTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        h0_val = backend.execute(result["h0"]).tolist()[0]
        h1_val = backend.execute(result["h1"]).tolist()[0]
        assert h0_val == pytest.approx(2.0, rel=1e-6)
        assert h1_val == pytest.approx(3.0, rel=1e-6)