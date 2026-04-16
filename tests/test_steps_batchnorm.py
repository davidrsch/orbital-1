"""Tests for BatchNormalization pipeline step translators."""

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


