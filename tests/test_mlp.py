"""Tests for MLP (neural network) support via Relu, Tanh, Sigmoid, Gemm translators."""

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


class TestReluTranslator:
    optimizer = Optimizer(enabled=False)

    def test_relu_single_positive(self):
        """Relu on positive values should be identity."""
        table = ibis.memtable({"input": [2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Relu(input)
            }
        """)
        variables = GraphVariables(table, model)
        translator = ReluTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        values = backend.execute(result).tolist()
        assert values == [2.0, 3.0]

    def test_relu_single_negative(self):
        """Relu on negative/zero values should return 0."""
        table = ibis.memtable({"input": [-1.0, -2.0, 0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Relu(input)
            }
        """)
        variables = GraphVariables(table, model)
        translator = ReluTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        values = backend.execute(result).tolist()
        assert values == [0.0, 0.0, 0.0]

    def test_relu_group(self):
        """Relu applied element-wise to a NumericVariablesGroup."""
        table = ibis.memtable({"h0": [2.0, -1.0], "h1": [-3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Relu(input)
            }
        """)
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup({
            "h0": table["h0"],
            "h1": table["h1"],
        })
        translator = ReluTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        h0_vals = backend.execute(result["h0"]).tolist()
        h1_vals = backend.execute(result["h1"]).tolist()
        assert h0_vals == [2.0, 0.0]
        assert h1_vals == [0.0, 4.0]

    def test_relu_invalid_input(self):
        """Relu should raise ValueError for non-numeric input."""
        table = ibis.memtable({"input": ["a", "b"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] input) => (string[N] output) {
                output = Relu(input)
            }
        """)
        variables = GraphVariables(table, model)
        translator = ReluTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        with pytest.raises(ValueError, match="Relu"):
            translator.process()


# ---------------------------------------------------------------------------
# Unit tests: TanhTranslator
# ---------------------------------------------------------------------------


class TestTanhTranslator:
    optimizer = Optimizer(enabled=False)

    def test_tanh_single(self):
        """Tanh of 0.0 is 0.0."""
        table = ibis.memtable({"input": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Tanh(input)
            }
        """)
        variables = GraphVariables(table, model)
        translator = TanhTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        value = backend.execute(result).tolist()[0]
        assert abs(value - math.tanh(0.0)) < 1e-9

    def test_tanh_group(self):
        """Tanh applied element-wise to each key in a NumericVariablesGroup."""
        table = ibis.memtable({"h0": [0.5, -0.5], "h1": [1.0, -1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Tanh(input)
            }
        """)
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup({
            "h0": table["h0"],
            "h1": table["h1"],
        })
        translator = TanhTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        h0_vals = backend.execute(result["h0"]).tolist()
        h1_vals = backend.execute(result["h1"]).tolist()
        expected_h0 = [math.tanh(0.5), math.tanh(-0.5)]
        expected_h1 = [math.tanh(1.0), math.tanh(-1.0)]
        for got, exp in zip(h0_vals, expected_h0):
            assert abs(got - exp) < 1e-9
        for got, exp in zip(h1_vals, expected_h1):
            assert abs(got - exp) < 1e-9

    def test_tanh_invalid_input(self):
        """Tanh should raise ValueError for non-numeric input."""
        table = ibis.memtable({"input": ["a", "b"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] input) => (string[N] output) {
                output = Tanh(input)
            }
        """)
        variables = GraphVariables(table, model)
        translator = TanhTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        with pytest.raises(ValueError, match="Tanh"):
            translator.process()


# ---------------------------------------------------------------------------
# Unit tests: SigmoidTranslator
# ---------------------------------------------------------------------------


class TestSigmoidTranslator:
    optimizer = Optimizer(enabled=False)

    def test_sigmoid_single(self):
        """Sigmoid of 0.0 is 0.5."""
        table = ibis.memtable({"input": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Sigmoid(input)
            }
        """)
        variables = GraphVariables(table, model)
        translator = SigmoidTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        value = backend.execute(result).tolist()[0]
        assert abs(value - 0.5) < 1e-9

    def test_sigmoid_group(self):
        """Sigmoid of 0.0 is 0.5 for every element of a group."""
        table = ibis.memtable({"h0": [0.0, 0.0], "h1": [0.0, 0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Sigmoid(input)
            }
        """)
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup({
            "h0": table["h0"],
            "h1": table["h1"],
        })
        translator = SigmoidTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        for key in ("h0", "h1"):
            vals = backend.execute(result[key]).tolist()
            for v in vals:
                assert abs(v - 0.5) < 1e-9

    def test_sigmoid_invalid_input(self):
        """Sigmoid should raise ValueError for non-numeric input."""
        table = ibis.memtable({"input": ["a", "b"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] input) => (string[N] output) {
                output = Sigmoid(input)
            }
        """)
        variables = GraphVariables(table, model)
        translator = SigmoidTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        with pytest.raises(ValueError, match="Sigmoid"):
            translator.process()


# ---------------------------------------------------------------------------
# Unit tests: GemmTranslator
# ---------------------------------------------------------------------------


class TestGemmTranslator:
    optimizer = Optimizer(enabled=False)

    def _make_gemm_graph(self, input_dim, output_dim, b_data, c_data, transB=1):
        """Create an ONNX GraphProto with a Gemm node and fixed initializers."""
        b_shape = [output_dim, input_dim] if transB == 1 else [input_dim, output_dim]
        b_tensor = helper.make_tensor("B", TensorProto.FLOAT, b_shape, b_data)
        c_tensor = helper.make_tensor("C", TensorProto.FLOAT, [output_dim], c_data)
        gemm_node = helper.make_node(
            "Gemm",
            inputs=["A", "B", "C"],
            outputs=["output"],
            transB=transB,
        )
        graph = helper.make_graph(
            [gemm_node],
            "gemm_graph",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, [None, input_dim])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, output_dim])],
            initializer=[b_tensor, c_tensor],
        )
        return graph

    def test_gemm_single_output_transB1(self):
        """Gemm 2→1 output, transB=1: 1.0*0.5 + 2.0*0.3 + 0.1 = 1.2."""
        # Table must have a column named "A" to match the graph input.
        # We then override variables["A"] with the desired group.
        table = ibis.memtable({"A": [0.0], "feature1": [1.0], "feature2": [2.0]})
        # B stored as (output_dim=1, input_dim=2): [0.5, 0.3]
        # C: [0.1]
        # Expected: 1.0*0.5 + 2.0*0.3 + 0.1 = 1.2
        graph = self._make_gemm_graph(
            input_dim=2,
            output_dim=1,
            b_data=[0.5, 0.3],
            c_data=[0.1],
            transB=1,
        )
        variables = GraphVariables(table, graph)
        variables["A"] = NumericVariablesGroup({
            "feature1": table["feature1"],
            "feature2": table["feature2"],
        })
        translator = GemmTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        value = backend.execute(result).tolist()[0]
        assert abs(value - 1.2) < 1e-6

    def test_gemm_multi_output_transB1(self):
        """Gemm 2→2 output with identity weight matrix (transB=1): output == input."""
        table = ibis.memtable({"A": [0.0], "feature1": [3.0], "feature2": [4.0]})
        # B stored as (output_dim=2, input_dim=2): identity [[1,0],[0,1]] flattened = [1,0,0,1]
        # C: [0.0, 0.0]
        graph = self._make_gemm_graph(
            input_dim=2,
            output_dim=2,
            b_data=[1.0, 0.0, 0.0, 1.0],
            c_data=[0.0, 0.0],
            transB=1,
        )
        variables = GraphVariables(table, graph)
        variables["A"] = NumericVariablesGroup({
            "feature1": table["feature1"],
            "feature2": table["feature2"],
        })
        translator = GemmTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        backend = ibis.duckdb.connect()
        out0 = backend.execute(result["out_0"]).tolist()[0]
        out1 = backend.execute(result["out_1"]).tolist()[0]
        assert abs(out0 - 3.0) < 1e-6
        assert abs(out1 - 4.0) < 1e-6


# ---------------------------------------------------------------------------
# End-to-end tests: full MLP pipelines → SQL
# ---------------------------------------------------------------------------


class TestMLPEndToEnd:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.X = pd.DataFrame({
            "f1": rng.standard_normal(50),
            "f2": rng.standard_normal(50),
            "f3": rng.standard_normal(50),
        })
        self.y_reg = self.X["f1"] * 2.0 + self.X["f2"] * -1.0 + 0.5
        self.y_bin = (self.y_reg > 0).astype(int)
        self.y_multi = pd.cut(self.y_reg, bins=3, labels=[0, 1, 2]).astype(int)

        self.features = {
            "f1": types.DoubleColumnType(),
            "f2": types.DoubleColumnType(),
            "f3": types.DoubleColumnType(),
        }

    def _validate(self, pipeline, X, y, is_classification):
        import duckdb
        conn = duckdb.connect(":memory:")
        parsed = parse_pipeline(pipeline, self.features)
        sql = orbital.export_sql("data", parsed, dialect="duckdb")
        sklearn_pred = pipeline.predict(X)
        sql_results = execute_sql(sql, conn, "duckdb", X)
        if is_classification:
            sql_pred = sql_results["output_label"].astype(int).values
            np.testing.assert_array_equal(sklearn_pred, sql_pred)
        else:
            sql_pred = sql_results["variable"].values
            np.testing.assert_allclose(sklearn_pred, sql_pred, rtol=1e-4, atol=1e-4)

    def test_mlp_regressor_relu_single_layer(self):
        """MLP regressor, single hidden layer, relu activation."""
        import duckdb
        conn = duckdb.connect(":memory:")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(10,),
                activation="relu",
                max_iter=200,
                random_state=42,
            )),
        ])
        pipeline.fit(self.X, self.y_reg)
        self._validate(pipeline, self.X, self.y_reg, is_classification=False)

    def test_mlp_regressor_relu_two_layers(self):
        """MLP regressor, two hidden layers, relu activation."""
        import duckdb
        conn = duckdb.connect(":memory:")
        pipeline = Pipeline([
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(10, 5),
                activation="relu",
                max_iter=200,
                random_state=42,
            )),
        ])
        pipeline.fit(self.X, self.y_reg)
        self._validate(pipeline, self.X, self.y_reg, is_classification=False)

    def test_mlp_regressor_tanh_single_layer(self):
        """MLP regressor, single hidden layer, tanh activation."""
        import duckdb
        conn = duckdb.connect(":memory:")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(8,),
                activation="tanh",
                max_iter=200,
                random_state=42,
            )),
        ])
        pipeline.fit(self.X, self.y_reg)
        self._validate(pipeline, self.X, self.y_reg, is_classification=False)

    def test_mlp_regressor_logistic_single_layer(self):
        """MLP regressor, single hidden layer, logistic (=sigmoid) activation."""
        import duckdb
        conn = duckdb.connect(":memory:")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(8,),
                activation="logistic",
                max_iter=200,
                random_state=42,
            )),
        ])
        pipeline.fit(self.X, self.y_reg)
        self._validate(pipeline, self.X, self.y_reg, is_classification=False)

    def test_mlp_classifier_binary_relu(self):
        """MLP binary classifier, relu activation."""
        import duckdb
        conn = duckdb.connect(":memory:")
        pipeline = Pipeline([
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(10,),
                activation="relu",
                max_iter=200,
                random_state=42,
            )),
        ])
        pipeline.fit(self.X, self.y_bin)
        self._validate(pipeline, self.X, self.y_bin, is_classification=True)

    def test_mlp_classifier_multiclass_relu(self):
        """MLP multiclass classifier (3 classes), relu activation."""
        import duckdb
        conn = duckdb.connect(":memory:")
        pipeline = Pipeline([
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(10,),
                activation="relu",
                max_iter=500,
                random_state=42,
            )),
        ])
        pipeline.fit(self.X, self.y_multi)
        self._validate(pipeline, self.X, self.y_multi, is_classification=True)

    def test_mlp_classifier_binary_tanh(self):
        """MLP binary classifier, tanh activation."""
        import duckdb
        conn = duckdb.connect(":memory:")
        pipeline = Pipeline([
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(8,),
                activation="tanh",
                max_iter=200,
                random_state=42,
            )),
        ])
        pipeline.fit(self.X, self.y_bin)
        self._validate(pipeline, self.X, self.y_bin, is_classification=True)

    def test_mlp_pipeline_scaler_then_mlp(self):
        """Pipeline with StandardScaler followed by MLP regressor."""
        import duckdb
        conn = duckdb.connect(":memory:")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(5,),
                activation="relu",
                max_iter=200,
                random_state=42,
            )),
        ])
        pipeline.fit(self.X, self.y_reg)
        self._validate(pipeline, self.X, self.y_reg, is_classification=False)

# ---------------------------------------------------------------------------
# Unit tests: ExpTranslator
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
        table = ibis.memtable({"h0": [3.0], "h1": [4.0]})
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
        table = ibis.memtable({"h0": [1.0], "h1": [1.0]})
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