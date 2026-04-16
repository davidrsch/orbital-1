"""Tests for GroupNorm MeanVarianceNorm RMSNorm pipeline step translators."""

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

    def test_axes_1_accepted(self):
        """axes=[1] must not raise NotImplementedError.

        Regression test: previously any explicit axes value raised
        NotImplementedError; axes=[1] (feature-axis normalisation for 2-D
        tabular input) is now accepted as equivalent to the default behaviour.
        """
        table = ibis.memtable({"a": [1.0], "b": [3.0]})
        node = helper.make_node(
            "MeanVarianceNormalization",
            inputs=["x"],
            outputs=["output"],
            axes=[1],
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
        # Must not raise — axes=[1] is the supported normalisation axis for 2-D input.
        MeanVarianceNormalizationTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)


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

