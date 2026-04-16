"""Tests for Gelu PRelu Softplus Mish LogSigmoid pipeline step translators."""

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

class TestGeluTranslator:
    """Tests for GeluTranslator (ONNX opset 20+ Gelu op)."""


    def test_gelu_registered(self):
        from orbital.translation.steps.gelu import GeluTranslator
        assert TRANSLATORS.get("Gelu") is GeluTranslator

    def test_gelu_none_zero(self):
        """Gelu(0) = 0 for approximate='none'."""
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Gelu <approximate: string = "none"> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.gelu import GeluTranslator
        t = GeluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - 0.0) < 1e-6

    def test_gelu_tanh_positive(self):
        """Gelu(2.0) with approximate='tanh' should be close to the true GELU value."""
        import math
        table = ibis.memtable({"x": [2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Gelu <approximate: string = "tanh"> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.gelu import GeluTranslator
        t = GeluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        # tanh approx: 0.5 * 2 * (1 + tanh(sqrt(2/pi) * (2 + 0.044715*8)))
        c = math.sqrt(2.0 / math.pi)
        expected = 0.5 * 2.0 * (1.0 + math.tanh(c * (2.0 + 0.044715 * 8.0)))
        assert abs(val - expected) < 1e-4

    def test_gelu_none_positive(self):
        """Gelu(1.0) with approximate='none' ~= 0.8413."""
        import math
        table = ibis.memtable({"x": [1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Gelu <approximate: string = "none"> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.gelu import GeluTranslator
        t = GeluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        # True GELU(1.0)  0.8413
        assert abs(val - 0.8413) < 0.01

    def test_gelu_group(self):
        """Gelu applied element-wise over a VariablesGroup."""
        table = ibis.memtable({"h0": [0.0, 1.0], "h1": [2.0, -1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Gelu <approximate: string = "none"> (x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"h0": table["h0"], "h1": table["h1"]})
        from orbital.translation.steps.gelu import GeluTranslator
        t = GeluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)



class TestPReluTranslator:
    """Tests for PreluTranslator."""


    def _make_prelu_graph(self, slope_val: float):
        return onnx.parser.parse_graph(f"""
            agraph (float[N] x) => (float[N] output)
            <float[1] slope = {{{slope_val}}}>
            {{
                output = PRelu(x, slope)
            }}
        """)

    def test_prelu_registered(self):
        from orbital.translation.steps.prelu import PReluTranslator
        assert TRANSLATORS.get("PRelu") is PReluTranslator

    def test_prelu_positive_unchanged(self):
        """PRelu of positive values = x (slope doesn't matter)."""
        table = ibis.memtable({"x": [1.0, 2.0, 3.0]})
        model = self._make_prelu_graph(0.25)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.prelu import PReluTranslator
        t = PReluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [1.0, 2.0, 3.0]

    def test_prelu_negative_scaled(self):
        """PRelu of negative values = slope * x."""
        table = ibis.memtable({"x": [-2.0, -4.0]})
        model = self._make_prelu_graph(0.5)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.prelu import PReluTranslator
        t = PReluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert abs(result[0] - (-1.0)) < 1e-9
        assert abs(result[1] - (-2.0)) < 1e-9



class TestSoftplusTranslator:
    """Tests for SoftplusTranslator."""


    def test_softplus_registered(self):
        from orbital.translation.steps.softplus import SoftplusTranslator
        assert TRANSLATORS.get("Softplus") is SoftplusTranslator

    def test_softplus_zero(self):
        """Softplus(0) = ln(2)  0.6931."""
        import math
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Softplus(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.softplus import SoftplusTranslator
        t = SoftplusTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - math.log(2.0)) < 1e-9

    def test_softplus_positive(self):
        """Softplus(3) = ln(1+e^3)  3.049."""
        import math
        table = ibis.memtable({"x": [3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Softplus(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.softplus import SoftplusTranslator
        t = SoftplusTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - math.log(1 + math.exp(3.0))) < 1e-9



class TestMishTranslator:
    """Tests for MishTranslator."""


    def test_mish_registered(self):
        from orbital.translation.steps.mish import MishTranslator
        assert TRANSLATORS.get("Mish") is MishTranslator

    def test_mish_zero(self):
        """Mish(0) = 0 * tanh(ln(2))  0."""
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Mish(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.mish import MishTranslator
        t = MishTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - 0.0) < 1e-9

    def test_mish_positive(self):
        """Mish(1) = 1 * tanh(ln(1+e))  0.865."""
        import math
        table = ibis.memtable({"x": [1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Mish(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.mish import MishTranslator
        t = MishTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        expected = 1.0 * math.tanh(math.log(1.0 + math.e))
        assert abs(val - expected) < 1e-9



class TestLogSigmoidTranslator:
    """Tests for LogSigmoidTranslator."""


    def test_logsigmoid_registered(self):
        from orbital.translation.steps.logsigmoid import LogSigmoidTranslator
        assert TRANSLATORS.get("LogSigmoid") is LogSigmoidTranslator

    def test_logsigmoid_zero(self):
        """LogSigmoid(0) = ln(0.5)  -0.6931."""
        import math
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = LogSigmoid(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.logsigmoid import LogSigmoidTranslator
        t = LogSigmoidTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - math.log(0.5)) < 1e-9

    def test_logsigmoid_large_positive(self):
        """LogSigmoid(large positive)  0."""
        table = ibis.memtable({"x": [100.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = LogSigmoid(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.logsigmoid import LogSigmoidTranslator
        t = LogSigmoidTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - 0.0) < 1e-4



