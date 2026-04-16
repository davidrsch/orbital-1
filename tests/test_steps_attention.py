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

class TestAttentionTranslator:
    """Tests for the ONNX Attention (com.microsoft fused-QKV) translator."""

    backend = ibis.duckdb.connect()

    def _make_attention_graph(self, T, I_size, H, num_heads, w_flat, bias_flat=None):
        """Build a minimal ONNX Attention graph."""
        inputs = ["X", "W", "B"] if bias_flat is not None else ["X", "W"]
        node = helper.make_node(
            "Attention",
            inputs=inputs,
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
        )
        inits = [
            helper.make_tensor("W", TensorProto.FLOAT, [I_size, 3 * H], w_flat),
        ]
        if bias_flat is not None:
            inits.append(
                helper.make_tensor("B", TensorProto.FLOAT, [3 * H], bias_flat)
            )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I_size])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, T * H])],
            inits,
        )
        return graph

    def test_attention_registered(self):
        from orbital.translation.steps.attention import AttentionTranslator
        assert TRANSLATORS.get("Attention") is AttentionTranslator

    def test_attention_identity_weight_single_timestep(self):
        """Single timestep: identity QKV weights, no bias → softmax of one score = 1.0."""
        from orbital.translation.steps.attention import AttentionTranslator

        # T=1, I=2, H=2, num_heads=1, head_dim=2
        # W = identity-like: just a 2x6 matrix (first two cols = Q, next = K, last = V)
        # Use all-zero weights for simplicity → output should be all zeros
        T, I_size, H, num_heads = 1, 2, 2, 1
        w_flat = [0.0] * (I_size * 3 * H)

        graph = self._make_attention_graph(T, I_size, H, num_heads, w_flat)
        cols = {"x0": [1.0], "x1": [2.0]}
        table = ibis.memtable(cols)
        x_group = ValueVariablesGroup({k: table[k] for k in cols})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = x_group

        AttentionTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        # T * H = 2 output columns
        assert len(result) == T * H
        # all weights zero → all projections zero → output zero
        for expr in result.values():
            val = self.backend.execute(expr).item()
            assert abs(val) < 1e-9

    def test_attention_multi_head_output_shape(self):
        """Multi-head: output group has T * H entries."""
        from orbital.translation.steps.attention import AttentionTranslator

        T, I_size, H, num_heads = 2, 4, 4, 2  # head_dim = 2
        import random
        random.seed(42)
        w_flat = [random.gauss(0, 0.1) for _ in range(I_size * 3 * H)]

        graph = self._make_attention_graph(T, I_size, H, num_heads, w_flat)
        cols = {f"x{i}": [float(i) / 10] for i in range(T * I_size)}
        table = ibis.memtable(cols)
        x_group = ValueVariablesGroup({k: table[k] for k in cols})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = x_group

        AttentionTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == T * H

    def test_attention_mask_index_raises(self):
        """Passing mask_index (input[3]) must raise NotImplementedError."""
        from orbital.translation.steps.attention import AttentionTranslator

        T, I_size, H, num_heads = 1, 2, 2, 1
        w_flat = [0.0] * (I_size * 3 * H)
        mask_tensor = helper.make_tensor("M", TensorProto.INT32, [1], [1])
        node = helper.make_node(
            "Attention",
            inputs=["X", "W", "", "M"],
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
        )
        inits = [
            helper.make_tensor("W", TensorProto.FLOAT, [I_size, 3 * H], w_flat),
            mask_tensor,
        ]
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I_size])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, T * H])],
            inits,
        )
        cols = {"x0": [1.0], "x1": [2.0]}
        table = ibis.memtable(cols)
        x_group = ValueVariablesGroup({k: table[k] for k in cols})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = x_group
        with pytest.raises(NotImplementedError, match="mask_index"):
            AttentionTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()

    def test_attention_past_input_raises_not_implemented(self):
        """Attention: providing a past KV-state input (input[4]) must raise NotImplementedError."""
        from orbital.translation.steps.attention import AttentionTranslator

        T, I_size, H, num_heads = 1, 2, 2, 1
        w_flat = [0.0] * (I_size * 3 * H)
        # input[2]=bias absent, input[3]=mask_index absent, input[4]=past present
        past_tensor = helper.make_tensor("past", TensorProto.FLOAT, [1], [0.0])
        node = helper.make_node(
            "Attention",
            inputs=["X", "W", "", "", "past"],
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
        )
        inits = [
            helper.make_tensor("W", TensorProto.FLOAT, [I_size, 3 * H], w_flat),
            past_tensor,
        ]
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I_size])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, T * H])],
            inits,
        )
        cols = {"x0": [1.0], "x1": [2.0]}
        table = ibis.memtable(cols)
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({k: table[k] for k in cols})

        with pytest.raises(NotImplementedError, match=r"past \(input\[4\]\)"):
            AttentionTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()


# ---------------------------------------------------------------------------
# Unit tests: MeanVarianceNormalizationTranslator
# ---------------------------------------------------------------------------


