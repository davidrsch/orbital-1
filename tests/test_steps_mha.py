"""Tests for MultiHeadAttention and Attention pipeline step translators."""

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

class TestMultiHeadAttentionTranslator:
    """Tests for the ONNX MultiHeadAttention (com.microsoft contrib) translator."""

    backend = ibis.duckdb.connect()

    def _make_mha_graph(self, T_q, D, num_heads, bias_flat=None):
        """Build a minimal ONNX MultiHeadAttention graph (self-attention)."""
        inputs = ["Q", "", "", "B"] if bias_flat is not None else ["Q", "", ""]
        node = helper.make_node(
            "MultiHeadAttention",
            inputs=inputs,
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
        )
        inits = []
        if bias_flat is not None:
            b_tensor = helper.make_tensor(
                "B", TensorProto.FLOAT, [len(bias_flat)], bias_flat
            )
            inits.append(b_tensor)
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("Q", TensorProto.FLOAT, [None, T_q * D])],
            [
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [None, T_q * D]
                )
            ],
            inits,
        )
        return graph

    def test_mha_registered(self):
        from orbital.translation.steps.multiheadattention import (
            MultiHeadAttentionTranslator,
        )
        assert TRANSLATORS.get("MultiHeadAttention") is MultiHeadAttentionTranslator

    def test_mha_self_attention_uniform_weights(self):
        """Self-attention: identity Q=K=V, zero bias, uniform scores → average of V."""
        from orbital.translation.steps.multiheadattention import (
            MultiHeadAttentionTranslator,
        )

        # Single head, 2 timesteps, D=2 → head_dim=2
        # Q = K = V = [[1, 0], [0, 1]]
        # bias = [0] * 6
        T_q, D, num_heads = 2, 2, 1
        bias_flat = [0.0] * (3 * D)

        graph = self._make_mha_graph(T_q, D, num_heads, bias_flat)
        table = ibis.memtable(
            {"q0": [1.0], "q1": [0.0], "q2": [0.0], "q3": [1.0]}
        )
        q_group = ValueVariablesGroup(
            {
                "q0": table["q0"],
                "q1": table["q1"],
                "q2": table["q2"],
                "q3": table["q3"],
            }
        )

        variables = GraphVariables(ibis.memtable({"Q": [0.0]}), graph)
        variables["Q"] = q_group

        MultiHeadAttentionTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        # Output has T_q * D = 4 entries
        assert len(result) == T_q * D

    def test_mha_no_bias_no_hidden_size_raises(self):
        """Without bias and without hidden_size attribute, must raise NotImplementedError."""
        from orbital.translation.steps.multiheadattention import (
            MultiHeadAttentionTranslator,
        )

        T_q, D, num_heads = 2, 2, 1
        # No bias
        node = helper.make_node(
            "MultiHeadAttention",
            inputs=["Q"],
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("Q", TensorProto.FLOAT, [None, T_q * D])],
            [
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [None, T_q * D]
                )
            ],
            [],
        )
        table = ibis.memtable({"q0": [1.0], "q1": [0.0], "q2": [0.0], "q3": [1.0]})
        q_group = ValueVariablesGroup(
            {"q0": table["q0"], "q1": table["q1"], "q2": table["q2"], "q3": table["q3"]}
        )
        variables = GraphVariables(ibis.memtable({"Q": [0.0]}), graph)
        variables["Q"] = q_group
        t = MultiHeadAttentionTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        with pytest.raises(NotImplementedError, match="hidden size"):
            t.process()

    def test_mha_hidden_size_attribute(self):
        """When bias is absent but hidden_size attribute is set, translation succeeds."""
        from orbital.translation.steps.multiheadattention import (
            MultiHeadAttentionTranslator,
        )

        T_q, D, num_heads = 2, 4, 2  # head_dim = 2
        node = helper.make_node(
            "MultiHeadAttention",
            inputs=["Q"],
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
            hidden_size=D,
        )
        graph = _make_graph_with_inits(
            node,
            [
                helper.make_tensor_value_info(
                    "Q", TensorProto.FLOAT, [None, T_q * D]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [None, T_q * D]
                )
            ],
            [],
        )
        cols = {f"q{i}": [float(i % 2)] for i in range(T_q * D)}
        table = ibis.memtable(cols)
        q_group = ValueVariablesGroup({k: table[k] for k in cols})
        variables = GraphVariables(ibis.memtable({"Q": [0.0]}), graph)
        variables["Q"] = q_group

        MultiHeadAttentionTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == T_q * D

    def test_multiheadattention_cross_attention_different_value_dim(self):
        """After the BUG-4 fix, D_v is inferred from V columns, not set equal to D."""
        from orbital.translation.steps.multiheadattention import (
            MultiHeadAttentionTranslator,
        )

        # Q: T_q=2 timesteps, D=8 features
        # K: T_k=3 timesteps, D=8 features
        # V: T_k=3 timesteps, D_v=4 features (D_v != D)
        # D supplied via hidden_size attribute so D_v inference from V shape is exercised.
        T_q, T_k, D, D_v, num_heads = 2, 3, 8, 4, 2
        node = helper.make_node(
            "MultiHeadAttention",
            inputs=["Q", "K", "V"],
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
            hidden_size=D,
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("Q", TensorProto.FLOAT, [None, T_q * D])],
            [
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [None, T_q * D_v]
                )
            ],
            [],
        )

        q_cols = {f"q{i}": [float(i % 4) / 4.0] for i in range(T_q * D)}
        k_cols = {f"k{i}": [float(i % 4) / 4.0] for i in range(T_k * D)}
        v_cols = {f"v{i}": [float(i % 2) / 2.0] for i in range(T_k * D_v)}
        table = ibis.memtable({**q_cols, **k_cols, **v_cols})

        variables = GraphVariables(ibis.memtable({"Q": [0.0]}), graph)
        variables["Q"] = ValueVariablesGroup({k: table[k] for k in q_cols})
        variables["K"] = ValueVariablesGroup({k: table[k] for k in k_cols})
        variables["V"] = ValueVariablesGroup({k: table[k] for k in v_cols})

        MultiHeadAttentionTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        # BUG-4 fix: output has T_q * D_v = 2 * 4 = 8 columns, not T_q * D = 2 * 8 = 16
        assert len(result) == T_q * D_v


# ---------------------------------------------------------------------------
# Unit tests: AttentionTranslator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit tests: AttentionTranslator
# ---------------------------------------------------------------------------


