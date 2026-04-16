"""Tests for RNN forward and basic pipeline step translators."""

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


