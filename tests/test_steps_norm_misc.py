"""Tests for misc normalization (LpNorm, LRN, parity) pipeline step translators."""

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



class TestBatchNormActivationParity:
    """Parity test: BatchNormalization output piped into Relu matches numpy reference.

    Covers GitHub issue #36 — ensures that composing BatchNorm + a standalone
    Activation node produces numerically identical results to the manually
    computed reference formula.
    """

    def _bn_relu_reference(self, x, scale, bias, mean, var, eps=1e-5):
        """Compute BatchNorm(x) then ReLU per-element using numpy."""
        import numpy as np
        x = np.array(x, dtype=float)
        scale = np.array(scale, dtype=float)
        bias = np.array(bias, dtype=float)
        mean = np.array(mean, dtype=float)
        var = np.array(var, dtype=float)
        bn = (x - mean) / np.sqrt(var + eps) * scale + bias
        return np.maximum(bn, 0.0)

    def test_batchnorm_relu_single_feature(self):
        """BatchNorm(x) + Relu: negative normalised values become 0."""
        import math
        import numpy as np

        xs = [0.5, 2.0, 5.0]
        scale_v, bias_v, mean_v, var_v = [1.5], [0.0], [2.0], [1.0]

        table = ibis.memtable({"x": xs})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <float[1] scale = {1.5}, float[1] bias = {0.0},
             float[1] mean  = {2.0}, float[1] var   = {1.0}>
            {
                bn_out = BatchNormalization(x, scale, bias, mean, var)
                output = Relu(bn_out)
            }
        """)
        variables = GraphVariables(table, model)

        from orbital.translation.steps.batchnorm import BatchNormalizationTranslator
        from orbital.translation.steps.relu import ReluTranslator

        BatchNormalizationTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        ReluTranslator(
            table, model.node[1], variables, self.optimizer, TranslationOptions()
        ).process()

        backend = ibis.duckdb.connect()
        got = list(backend.execute(variables.peek_variable("output")))
        expected = list(self._bn_relu_reference(xs, scale_v, bias_v, mean_v, var_v))
        for g, e in zip(got, expected):
            assert abs(g - e) < 1e-6, f"got {g}, expected {e}"

    def test_batchnorm_relu_multi_feature(self):
        """BatchNorm(columns) + Relu applied across multiple features."""
        import numpy as np

        a_vals = [1.0, 3.0]
        b_vals = [5.0, 7.0]
        scale_v = [2.0, 0.5]
        bias_v = [-1.0, 1.0]
        mean_v = [2.0, 6.0]
        var_v = [1.0, 1.0]

        table = ibis.memtable({"a": a_vals, "b": b_vals})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <float[2] scale = {2.0, 0.5}, float[2] bias  = {-1.0, 1.0},
             float[2] mean  = {2.0, 6.0}, float[2] var   = {1.0, 1.0}>
            {
                bn_out = BatchNormalization(x, scale, bias, mean, var)
                output = Relu(bn_out)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})

        from orbital.translation.steps.batchnorm import BatchNormalizationTranslator
        from orbital.translation.steps.relu import ReluTranslator

        BatchNormalizationTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        ReluTranslator(
            table, model.node[1], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        got_a = list(backend.execute(result["a"]))
        got_b = list(backend.execute(result["b"]))

        ref_a = list(self._bn_relu_reference(a_vals, [scale_v[0]], [bias_v[0]], [mean_v[0]], [var_v[0]]))
        ref_b = list(self._bn_relu_reference(b_vals, [scale_v[1]], [bias_v[1]], [mean_v[1]], [var_v[1]]))

        for g, e in zip(got_a, ref_a):
            assert abs(g - e) < 1e-6, f"col a: got {g}, expected {e}"
        for g, e in zip(got_b, ref_b):
            assert abs(g - e) < 1e-6, f"col b: got {g}, expected {e}"


# ---------------------------------------------------------------------------
# Unit tests: LpNormalizationTranslator (Issue #35)
# ---------------------------------------------------------------------------


class TestLpNormalizationTranslator:
    """Tests for LpNormalizationTranslator."""

    def test_l2norm_registered(self):
        from orbital.translation.steps.lpnormalization import LpNormalizationTranslator
        assert TRANSLATORS.get("LpNormalization") is LpNormalizationTranslator

    def test_l2norm_unit_vector(self):
        """LpNorm p=2 of [3, 4]: norm=5; result=[3/5, 4/5]."""
        table = ibis.memtable({"a": [3.0], "b": [4.0]})
        node = helper.make_node(
            "LpNormalization", inputs=["x"], outputs=["output"], p=2, axis=-1
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
            [],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.lpnormalization import LpNormalizationTranslator
        LpNormalizationTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        assert abs(backend.execute(result["a"])[0] - 3.0 / 5.0) < 1e-6
        assert abs(backend.execute(result["b"])[0] - 4.0 / 5.0) < 1e-6

    def test_l1norm_correctness(self):
        """LpNorm p=1 of [1, -3]: norm=4; result=[1/4, -3/4]."""
        table = ibis.memtable({"a": [1.0], "b": [-3.0]})
        node = helper.make_node(
            "LpNormalization", inputs=["x"], outputs=["output"], p=1, axis=-1
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
            [],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.lpnormalization import LpNormalizationTranslator
        LpNormalizationTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        assert abs(backend.execute(result["a"])[0] - 1.0 / 4.0) < 1e-6
        assert abs(backend.execute(result["b"])[0] - (-3.0 / 4.0)) < 1e-6

    def test_unsupported_p_raises(self):
        """LpNorm with p=3 raises NotImplementedError."""
        table = ibis.memtable({"a": [1.0], "b": [2.0]})
        node = helper.make_node(
            "LpNormalization", inputs=["x"], outputs=["output"], p=3, axis=-1
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
            [],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.lpnormalization import LpNormalizationTranslator
        with pytest.raises(NotImplementedError, match="p=3"):
            LpNormalizationTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()


# ---------------------------------------------------------------------------
# Unit tests: LRNTranslator (Issue #35)
# ---------------------------------------------------------------------------


class TestLRNTranslator:
    """Tests for LRNTranslator."""

    def test_lrn_registered(self):
        from orbital.translation.steps.localresponsenormalization import LRNTranslator
        assert TRANSLATORS.get("LRN") is LRNTranslator

    def test_lrn_default_params(self):
        """LRN with size=3, default alpha/beta/k against numpy reference."""
        import math
        import numpy as np

        # 4 features; size=3 window
        a, b, c, d = 1.0, 2.0, 3.0, 4.0
        size, alpha, beta, k = 3, 0.0001, 0.75, 1.0

        table = ibis.memtable({"a": [a], "b": [b], "c": [c], "d": [d]})
        node = helper.make_node(
            "LRN",
            inputs=["x"],
            outputs=["output"],
            size=size,
            alpha=alpha,
            beta=beta,
            bias=k,
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 4])],
            [],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = NumericVariablesGroup({
            "a": table["a"], "b": table["b"],
            "c": table["c"], "d": table["d"],
        })
        from orbital.translation.steps.localresponsenormalization import LRNTranslator
        LRNTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()

        xs = np.array([a, b, c, d])
        # numpy reference
        def lrn_ref(xi, lo, hi):
            sq = np.sum(xs[lo:hi+1] ** 2)
            return xi / (k + (alpha / size) * sq) ** beta

        n = len(xs)
        half_l = (size - 1) // 2
        half_r = math.ceil((size - 1) / 2)
        expected = {
            "a": lrn_ref(a, max(0, 0 - half_l), min(n - 1, 0 + half_r)),
            "b": lrn_ref(b, max(0, 1 - half_l), min(n - 1, 1 + half_r)),
            "c": lrn_ref(c, max(0, 2 - half_l), min(n - 1, 2 + half_r)),
            "d": lrn_ref(d, max(0, 3 - half_l), min(n - 1, 3 + half_r)),
        }
        for col, exp in expected.items():
            got = backend.execute(result[col])[0]
            assert abs(got - exp) < 1e-6, f"col {col}: got {got}, expected {exp}"


# ---------------------------------------------------------------------------
# Unit tests: LogTranslator
# ---------------------------------------------------------------------------   


class TestSkipLayerNormalizationTranslator:
    """Tests for SkipLayerNormalizationTranslator."""

    def test_skiplayernorm_registered(self):
        from orbital.translation.steps.skip_layer_norm import SkipLayerNormalizationTranslator
        assert TRANSLATORS.get("SkipLayerNormalization") is SkipLayerNormalizationTranslator

    def test_skiplayernorm_basic(self):
        """SkipLayerNorm(input, skip, gamma, beta) == LayerNorm(input + skip)."""
        import math
        n = 3  # hidden size
        row_input = [0.1, -0.5, 0.8]
        row_skip = [0.2, 0.3, -0.1]
        gamma_vals = [1.0, 1.0, 1.0]
        beta_vals = [0.0, 0.0, 0.0]
        epsilon = 1e-12

        combined = [a + b for a, b in zip(row_input, row_skip)]
        mean_c = sum(combined) / n
        var_c = sum((v - mean_c) ** 2 for v in combined) / n
        std_c = math.sqrt(var_c + epsilon)
        expected = [(v - mean_c) / std_c * g + b for v, g, b in zip(combined, gamma_vals, beta_vals)]

        table_data = {f"i{j}": [row_input[j]] for j in range(n)}
        table_data.update({f"s{j}": [row_skip[j]] for j in range(n)})
        table = ibis.memtable(table_data)

        gamma_tensor = helper.make_tensor("gamma", TensorProto.FLOAT, [n], gamma_vals)
        beta_tensor = helper.make_tensor("beta", TensorProto.FLOAT, [n], beta_vals)

        node = helper.make_node(
            "SkipLayerNormalization",
            inputs=["input", "skip", "gamma", "beta"],
            outputs=["output"],
            domain="com.microsoft",
            epsilon=epsilon,
        )
        graph = helper.make_graph(
            [node],
            "test_graph",
            [
                helper.make_tensor_value_info("input", TensorProto.FLOAT, [n]),
                helper.make_tensor_value_info("skip", TensorProto.FLOAT, [n]),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [n])],
            initializer=[gamma_tensor, beta_tensor],
        )

        from orbital.translation.steps.skip_layer_norm import SkipLayerNormalizationTranslator
        # GraphVariables needs a table with column names matching graph input names.
        dummy = ibis.memtable({"input": [1.0], "skip": [1.0]})
        variables = GraphVariables(dummy, graph)
        variables["input"] = NumericVariablesGroup({f"i{j}": table[f"i{j}"] for j in range(n)})
        variables["skip"] = NumericVariablesGroup({f"s{j}": table[f"s{j}"] for j in range(n)})

        t = SkipLayerNormalizationTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()

        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        for j, exp in enumerate(expected):
            val = list(backend.execute(list(result.values())[j]))[0]
            assert abs(val - exp) < 1e-6, f"Mismatch at index {j}: {val} != {exp}"
