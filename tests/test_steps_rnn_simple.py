"""Tests for SimpleRNN and Attention pipeline step translators."""

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

# ---------------------------------------------------------------------------
# Unit tests: LSTMTranslator
# ---------------------------------------------------------------------------



class TestRNNTranslator:
    """Tests for the ONNX RNN translator (forward-direction unrolling)."""


    def _make_rnn_graph(self, I, H, T, W_data, R_data, B_data=None):
        """Build a minimal ONNX RNN graph with Y_h output."""
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, H, I], W_data)
        R_tensor = helper.make_tensor("R", TensorProto.FLOAT, [1, H, H], R_data)
        inputs = ["X", "W", "R"]
        inits = [W_tensor, R_tensor]
        if B_data is not None:
            B_tensor = helper.make_tensor("B", TensorProto.FLOAT, [1, 2 * H], B_data)
            inputs.append("B")
            inits.append(B_tensor)
        node = helper.make_node(
            "RNN",
            inputs=inputs,
            outputs=["", "Y_h"],
            hidden_size=H,
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I])],
            [helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [None, H])],
            inits,
        )
        return graph

    def test_rnn_registered(self):
        from orbital.translation.steps.rnn import RNNTranslator
        assert TRANSLATORS.get("RNN") is RNNTranslator

    def test_rnn_zero_weights_zero_output(self):
        """All-zero weights and input: H_t = tanh(0) = 0."""
        from orbital.translation.steps.rnn import RNNTranslator

        I, H, T = 2, 2, 1
        W_data = [0.0] * (H * I)
        R_data = [0.0] * (H * H)

        graph = self._make_rnn_graph(I, H, T, W_data, R_data)
        table = ibis.memtable({"x0": [0.0], "x1": [0.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"], "x1": table["x1"]})

        RNNTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y_h")
        assert isinstance(result, ValueVariablesGroup)
        backend = ibis.duckdb.connect()
        for v in result.values():
            assert abs(backend.execute(v).tolist()[0]) < 1e-9

    def test_rnn_identity_weight_tanh(self):
        """W=identity, R=0, B=0, x=[1,0]: H_1 = tanh(1) at h=0, tanh(0) at h=1."""
        import math
        from orbital.translation.steps.rnn import RNNTranslator

        I, H, T = 2, 2, 1
        # W[h, i] = 1 if h==i else 0 (identity-like)
        W_data = [1.0, 0.0, 0.0, 1.0]  # H=2, I=2
        R_data = [0.0] * (H * H)

        graph = self._make_rnn_graph(I, H, T, W_data, R_data)
        table = ibis.memtable({"x0": [1.0], "x1": [0.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"], "x1": table["x1"]})

        RNNTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y_h")
        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        assert abs(vals[0] - math.tanh(1.0)) < 1e-6
        assert abs(vals[1] - math.tanh(0.0)) < 1e-9

    def test_rnn_direction_bidirectional_raises(self):
        """Bidirectional RNN with 1-direction weights raises ValueError."""
        from orbital.translation.steps.rnn import RNNTranslator

        I, H, T = 1, 1, 1
        W_data = [0.0] * (H * I)
        R_data = [0.0] * (H * H)
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, H, I], W_data)
        R_tensor = helper.make_tensor("R", TensorProto.FLOAT, [1, H, H], R_data)
        node = helper.make_node(
            "RNN",
            inputs=["X", "W", "R"],
            outputs=["", "Y_h"],
            hidden_size=H,
            direction="bidirectional",
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I])],
            [helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [None, H])],
            [W_tensor, R_tensor],
        )
        table = ibis.memtable({"x0": [1.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"]})
        t = RNNTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        with pytest.raises(ValueError, match="direction"):
            t.process()

    def test_relu_activation(self):
        """RNN with activations=['Relu']: positive input passes through, negative clamps to 0."""
        from orbital.translation.steps.rnn import RNNTranslator

        I, H, T = 2, 2, 1
        # Diagonal W: unit 0 reads x0, unit 1 reads x1
        W_data = [1.0, 0.0, 0.0, 1.0]  # shape [1, H=2, I=2]
        R_data = [0.0] * (H * H)
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, H, I], W_data)
        R_tensor = helper.make_tensor("R", TensorProto.FLOAT, [1, H, H], R_data)
        node = helper.make_node(
            "RNN",
            inputs=["X", "W", "R"],
            outputs=["", "Y_h"],
            hidden_size=H,
            activations=["Relu"],
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I])],
            [helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [None, H])],
            [W_tensor, R_tensor],
        )
        # x0=1.0 → h0 = max(0, 1.0) = 1.0
        # x1=-2.0 → h1 = max(0, -2.0) = 0.0
        table = ibis.memtable({"x0": [1.0], "x1": [-2.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"], "x1": table["x1"]})

        RNNTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y_h")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == H

        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        assert abs(vals[0] - 1.0) < 1e-9   # relu(1.0) = 1.0
        assert abs(vals[1] - 0.0) < 1e-9   # relu(-2.0) = 0.0

    def test_unsupported_activation_raises(self):
        """RNN with activations=['Elu'] raises NotImplementedError naming the activation."""
        from orbital.translation.steps.rnn import RNNTranslator

        I, H, T = 1, 1, 1
        W_data = [0.0] * (H * I)
        R_data = [0.0] * (H * H)
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, H, I], W_data)
        R_tensor = helper.make_tensor("R", TensorProto.FLOAT, [1, H, H], R_data)
        node = helper.make_node(
            "RNN",
            inputs=["X", "W", "R"],
            outputs=["", "Y_h"],
            hidden_size=H,
            activations=["Elu"],
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, T * I])],
            [helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [None, H])],
            [W_tensor, R_tensor],
        )
        table = ibis.memtable({"x0": [1.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"]})

        with pytest.raises(NotImplementedError, match="Elu"):
            RNNTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()


# ---------------------------------------------------------------------------
# Unit tests: MultiHeadAttentionTranslator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit tests: MultiHeadAttentionTranslator
# ---------------------------------------------------------------------------


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


