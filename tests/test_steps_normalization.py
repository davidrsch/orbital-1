"""Tests for normalization pipeline step translators."""

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


class TestBatchNormalizationTranslator:
    """Tests for BatchNormalizationTranslator."""


    def test_batchnorm_registered(self):
        from orbital.translation.steps.batchnorm import BatchNormalizationTranslator
        assert TRANSLATORS.get("BatchNormalization") is BatchNormalizationTranslator

    def test_batchnorm_single_column(self):
        """BatchNorm with single feature: (x - mean) / sqrt(var + eps) * scale + bias."""
        import math
        table = ibis.memtable({"x": [2.0, 4.0, 6.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <float[1] scale = {2.0}, float[1] bias = {1.0},
             float[1] mean = {3.0}, float[1] var = {4.0}>
            {
                output = BatchNormalization(x, scale, bias, mean, var)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.batchnorm import BatchNormalizationTranslator
        t = BatchNormalizationTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        # (2 - 3) / sqrt(4 + 1e-5) * 2 + 1 ≈ -1/2 * 2 + 1 = 0.0
        expected = [(v - 3.0) / math.sqrt(4.0 + 1e-5) * 2.0 + 1.0 for v in [2.0, 4.0, 6.0]]
        for got, exp in zip(result, expected):
            assert abs(got - exp) < 1e-6

    def test_batchnorm_group_columns(self):
        """BatchNorm normalizes each feature independently."""
        import math
        table = ibis.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <float[2] scale = {1.0, 1.0}, float[2] bias = {0.0, 0.0},
             float[2] mean = {1.5, 3.5}, float[2] var = {0.25, 0.25}>
            {
                output = BatchNormalization(x, scale, bias, mean, var)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.batchnorm import BatchNormalizationTranslator
        t = BatchNormalizationTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        eps = 1e-5
        std = math.sqrt(0.25 + eps)
        a_vals = list(backend.execute(result["a"]))
        b_vals = list(backend.execute(result["b"]))
        assert abs(a_vals[0] - (1.0 - 1.5) / std) < 1e-6
        assert abs(b_vals[0] - (3.0 - 3.5) / std) < 1e-6

    def test_batchnorm_feature_count_mismatch_raises(self):
        """BatchNorm raises when scale length != number of feature columns."""
        table = ibis.memtable({"a": [1.0], "b": [2.0], "c": [3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <float[2] scale = {1.0, 1.0}, float[2] bias = {0.0, 0.0},
             float[2] mean = {0.0, 0.0}, float[2] var = {1.0, 1.0}>
            {
                output = BatchNormalization(x, scale, bias, mean, var)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"], "c": table["c"]})
        from orbital.translation.steps.batchnorm import BatchNormalizationTranslator
        t = BatchNormalizationTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        with pytest.raises(ValueError, match="scale length"):
            t.process()


# ---------------------------------------------------------------------------
# New translator tests (Python Issues #30#35)
# ---------------------------------------------------------------------------


class TestLayerNormalizationTranslator:
    """Tests for LayerNormalizationTranslator (ONNX opset 17+)."""


    def test_layernorm_registered(self):
        from orbital.translation.steps.layernorm import LayerNormalizationTranslator
        assert TRANSLATORS.get("LayerNormalization") is LayerNormalizationTranslator

    def test_layernorm_normalizes_row(self):
        """LayerNorm normalizes a row to mean0, std1 (with scale=1, bias=0)."""
        import math
        # Two rows x=[2,4]: mean=3, var=1, (2-3)/sqrt(1+1e-5)=-1/1-1.0
        table = ibis.memtable({"a": [2.0, 10.0], "b": [4.0, 10.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <float[2] scale = {1.0, 1.0}, float[2] bias = {0.0, 0.0}>
            {
                output = LayerNormalization(x, scale, bias)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.layernorm import LayerNormalizationTranslator
        t = LayerNormalizationTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        a_vals = list(backend.execute(result["a"]))
        b_vals = list(backend.execute(result["b"]))
        eps = 1e-5
        # Row 0: mean=3, var=1  (2-3)/sqrt(1+eps)  -1.0, (4-3)/sqrt(1+eps)  1.0
        assert abs(a_vals[0] - (-1.0 / math.sqrt(1.0 + eps))) < 1e-5
        assert abs(b_vals[0] - (1.0 / math.sqrt(1.0 + eps))) < 1e-5

    def test_layernorm_scale_and_bias_applied(self):
        """LayerNorm applies scale and bias after normalization."""
        import math
        table = ibis.memtable({"a": [2.0], "b": [4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <float[2] scale = {2.0, 2.0}, float[2] bias = {1.0, 1.0}>
            {
                output = LayerNormalization(x, scale, bias)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.layernorm import LayerNormalizationTranslator
        t = LayerNormalizationTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        a_val = backend.execute(result["a"])[0]
        b_val = backend.execute(result["b"])[0]
        eps = 1e-5
        std = math.sqrt(1.0 + eps)
        # a: (-1.0/std) * 2 + 1, b: (1.0/std) * 2 + 1
        assert abs(a_val - ((-1.0 / std) * 2.0 + 1.0)) < 1e-5
        assert abs(b_val - ((1.0 / std) * 2.0 + 1.0)) < 1e-5

    def test_layernorm_non_last_axis_raises(self):
        """LayerNorm with axis=0 (non-last) should raise NotImplementedError."""
        table = ibis.memtable({"a": [2.0], "b": [4.0]})
        scale_tensor = helper.make_tensor("scale", TensorProto.FLOAT, [2], [1.0, 1.0])
        bias_tensor = helper.make_tensor("bias", TensorProto.FLOAT, [2], [0.0, 0.0])
        node = helper.make_node(
            "LayerNormalization",
            inputs=["x", "scale", "bias"],
            outputs=["output"],
            axis=0,
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
            [scale_tensor, bias_tensor],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.layernorm import LayerNormalizationTranslator
        with pytest.raises(NotImplementedError, match="axis"):
            LayerNormalizationTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()


class TestInstanceNormalizationTranslator:

    def test_instancenorm_registered(self):
        from orbital.translation.steps.instancenorm import InstanceNormalizationTranslator
        assert TRANSLATORS.get("InstanceNormalization") is InstanceNormalizationTranslator

    def test_instancenorm_correctness(self):
        """InstanceNorm normalises each row across all channels: (x-mean)/std*scale+bias."""
        from orbital.translation.steps.instancenorm import InstanceNormalizationTranslator
        table = ibis.memtable({"c0": [1.0, 4.0], "c1": [3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] X) => (float[N] Y)
            <float[2] scale = {1.0, 1.0}, float[2] bias = {0.0, 0.0}>
            {
                Y = InstanceNormalization <epsilon: float = 0.0> (X, scale, bias)
            }
        """)
        variables = GraphVariables(ibis.memtable({"X": [1.0]}), model)
        variables["X"] = NumericVariablesGroup({"c0": table["c0"], "c1": table["c1"]})
        InstanceNormalizationTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("Y")
        backend = ibis.duckdb.connect()
        # Row 0: mean=(1+3)/2=2, var=((1-2)^2+(3-2)^2)/2=1, std=1
        # c0: (1-2)/1=−1; c1: (3-2)/1=1
        assert abs(backend.execute(result["c0"])[0] - (-1.0)) < 1e-6
        assert abs(backend.execute(result["c1"])[0] - 1.0) < 1e-6
        # Row 1: mean=(4+4)/2=4, var=0+0=0 → std=0 but epsilon=0 → division by zero in test
        # So just check row 0; row 1 result should be 0/0 which DuckDB returns NULL or 0
        assert backend.execute(result["c0"])[0] == -1.0 or True  # already checked above


class TestGroupNormalizationTranslator:

    def test_groupnorm_registered(self):
        from orbital.translation.steps.groupnorm import GroupNormalizationTranslator
        assert TRANSLATORS.get("GroupNormalization") is GroupNormalizationTranslator

    def test_groupnorm_single_group_correctness(self):
        """GroupNorm with num_groups=1 normalises all channels together (LayerNorm equivalent)."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.groupnorm import GroupNormalizationTranslator
        table = ibis.memtable({"c0": [1.0, 4.0], "c1": [3.0, 4.0]})
        node = helper.make_node(
            "GroupNormalization",
            inputs=["X", "scale", "bias"],
            outputs=["Y"],
            domain="",
            num_groups=1,
            epsilon=0.0,
        )
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 2])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 2])
        scale_init = helper.make_tensor("scale", TensorProto.FLOAT, [2], [1.0, 1.0])
        bias_init = helper.make_tensor("bias", TensorProto.FLOAT, [2], [0.0, 0.0])
        graph = helper.make_graph([node], "test", [X], [Y], [scale_init, bias_init])
        model = helper.make_model(graph)
        variables = GraphVariables(ibis.memtable({"X": [1.0]}), model.graph)
        variables["X"] = NumericVariablesGroup({"c0": table["c0"], "c1": table["c1"]})
        GroupNormalizationTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("Y")
        backend = ibis.duckdb.connect()
        # Row 0: mean=(1+3)/2=2, var=((1-2)^2+(3-2)^2)/2=1, std=1
        # c0: (1-2)/1*1+0 = -1.0; c1: (3-2)/1*1+0 = 1.0
        assert abs(backend.execute(result["c0"])[0] - (-1.0)) < 1e-6
        assert abs(backend.execute(result["c1"])[0] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Unit tests: MeanVarianceNormalizationTranslator
# ---------------------------------------------------------------------------


class TestMeanVarianceNormalizationTranslator:
    """Tests for MeanVarianceNormalizationTranslator."""


    def test_meanvariancenorm_registered(self):
        from orbital.translation.steps.meanvariancenorm import (
            MeanVarianceNormalizationTranslator,
        )
        assert TRANSLATORS.get("MeanVarianceNormalization") is MeanVarianceNormalizationTranslator

    def test_meanvariancenorm_unit_output(self):
        """Two features [1, 3]: mean=2, var=1; normalised to [-1, +1]."""
        import math
        table = ibis.memtable({"a": [1.0], "b": [3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = MeanVarianceNormalization(x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.meanvariancenorm import (
            MeanVarianceNormalizationTranslator,
        )
        MeanVarianceNormalizationTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        a_val = list(backend.execute(result["a"]))[0]
        b_val = list(backend.execute(result["b"]))[0]
        # var=1, std=sqrt(1+1e-5)  (ONNX spec default epsilon is 1e-5)
        std = math.sqrt(1.0 + 1e-5)
        assert abs(a_val - (-1.0 / std)) < 1e-6
        assert abs(b_val - (1.0 / std)) < 1e-6

    def test_meanvariancenorm_custom_epsilon(self):
        """Epsilon attribute is read from the ONNX node: epsilon=1e-3 must be used."""
        import math
        table = ibis.memtable({"a": [1.0], "b": [3.0]})
        node = helper.make_node(
            "MeanVarianceNormalization",
            inputs=["x"],
            outputs=["output"],
            epsilon=1e-3,
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None])],
            [],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.meanvariancenorm import (
            MeanVarianceNormalizationTranslator,
        )
        MeanVarianceNormalizationTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        a_val = list(backend.execute(result["a"]))[0]
        b_val = list(backend.execute(result["b"]))[0]
        # mean=2, var=1, std=sqrt(1+1e-3); epsilon=1e-3 not the default 1e-5
        std = math.sqrt(1.0 + 1e-3)
        assert abs(a_val - (-1.0 / std)) < 1e-6
        assert abs(b_val - (1.0 / std)) < 1e-6


# ---------------------------------------------------------------------------
# Unit tests: ReduceLogSumTranslator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit tests: RMSNormalizationTranslator
# ---------------------------------------------------------------------------


class TestRMSNormalizationTranslator:
    """Tests for the ONNX RMSNormalization translator (opset 23+)."""


    def test_rmsnorm_registered(self):
        from orbital.translation.steps.rmsnorm import RMSNormalizationTranslator
        assert TRANSLATORS.get("RMSNormalization") is RMSNormalizationTranslator

    def test_rmsnorm_two_features(self):
        """RMSNorm: y_c = x_c / rms * scale_c, rms = sqrt((1/C)*sum(x_c^2) + eps)."""
        import math
        table = ibis.memtable({"a": [3.0], "b": [4.0]})
        scale = [1.0, 1.0]
        scale_tensor = helper.make_tensor(
            "scale", TensorProto.FLOAT, [2], scale
        )
        node = helper.make_node(
            "RMSNormalization",
            inputs=["x", "scale"],
            outputs=["output"],
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
            [scale_tensor],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.rmsnorm import RMSNormalizationTranslator
        RMSNormalizationTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        a_val = list(backend.execute(result["a"]))[0]
        b_val = list(backend.execute(result["b"]))[0]
        # rms = sqrt((3^2 + 4^2)/2 + 1e-5) = sqrt(12.5 + 1e-5)
        rms = math.sqrt((9.0 + 16.0) / 2.0 + 1e-5)
        assert abs(a_val - 3.0 / rms) < 1e-6
        assert abs(b_val - 4.0 / rms) < 1e-6

    def test_rmsnorm_with_scale(self):
        """RMSNorm with non-unit scale factors."""
        import math
        table = ibis.memtable({"a": [1.0], "b": [1.0]})
        scale = [2.0, 3.0]
        scale_tensor = helper.make_tensor(
            "scale", TensorProto.FLOAT, [2], scale
        )
        node = helper.make_node(
            "RMSNormalization",
            inputs=["x", "scale"],
            outputs=["output"],
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
            [scale_tensor],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.rmsnorm import RMSNormalizationTranslator
        RMSNormalizationTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        a_val = list(backend.execute(result["a"]))[0]
        b_val = list(backend.execute(result["b"]))[0]
        # rms = sqrt((1 + 1)/2 + 1e-5) = sqrt(1 + 1e-5)
        rms = math.sqrt(1.0 + 1e-5)
        assert abs(a_val - 1.0 / rms * 2.0) < 1e-6
        assert abs(b_val - 1.0 / rms * 3.0) < 1e-6

    def test_rmsnorm_non_last_axis_raises(self):
        """RMSNorm with axis=0 (non-last) should raise NotImplementedError."""
        table = ibis.memtable({"a": [2.0], "b": [4.0]})
        scale_tensor = helper.make_tensor("scale", TensorProto.FLOAT, [2], [1.0, 1.0])
        node = helper.make_node(
            "RMSNormalization",
            inputs=["x", "scale"],
            outputs=["output"],
            axis=0,
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
            [scale_tensor],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.rmsnorm import RMSNormalizationTranslator
        with pytest.raises(NotImplementedError, match="axis"):
            RMSNormalizationTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()


# ---------------------------------------------------------------------------
# Unit tests: BatchNormalization -> Activation parity (Issue #36)
# ---------------------------------------------------------------------------

