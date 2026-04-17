"""Tests for conv pipeline step translators."""

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


from conftest import make_graph_with_inits as _make_graph_with_inits

# ---------------------------------------------------------------------------
# Unit tests: ConvTranslator
# ---------------------------------------------------------------------------


class TestConvTranslator:
    """Tests for the ONNX Conv (1-D convolution) translator."""

    def test_conv_registered(self):
        from orbital.translation.steps.conv import ConvTranslator

        assert TRANSLATORS.get("Conv") is ConvTranslator

    def test_conv1d_single_filter_single_channel_valid(self):
        """1 filter, 1 in-channel, kernel=[1,1,1], no bias, valid pad.

        Input  : [1, 2, 3, 4]  shape [1, 1, 4] (NCHW, C=1, W=4)
        Kernel : shape [1, 1, 3] = [[[1, 1, 1]]]  (C_out=1, C_in=1, kW=3)
        Output : shape [1, 1, 2] (valid: W_out = 4-3+1 = 2)
          pos 0: 1+2+3 = 6
          pos 1: 2+3+4 = 9
        """
        from orbital.translation.steps.conv import ConvTranslator

        # Input group: C_in * W_in = 1*4 columns
        table = ibis.memtable({"x0": [1.0], "x1": [2.0], "x2": [3.0], "x3": [4.0]})

        W_data = [1.0, 1.0, 1.0]  # shape [1, 1, 3]
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, 1, 3], W_data)
        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"])
        node.attribute.extend(
            [
                helper.make_attribute("kernel_shape", [3]),
            ]
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 1, 4])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1, 2])],
            [W_tensor],
        )
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup(
            {f"x{i}": table[f"x{i}"] for i in range(4)}
        )

        translator = ConvTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("Y")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 2  # 1 filter  2 output positions

        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        assert abs(vals[0] - 6.0) < 1e-6
        assert abs(vals[1] - 9.0) < 1e-6

    def test_conv1d_with_bias(self):
        """Conv1D with a constant bias applied to every output position."""
        from orbital.translation.steps.conv import ConvTranslator

        table = ibis.memtable({"x0": [1.0], "x1": [2.0], "x2": [3.0]})

        W_data = [1.0, 0.0, 0.0]  # identity kernel: picks first element
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, 1, 3], W_data)
        B_tensor = helper.make_tensor("B", TensorProto.FLOAT, [1], [10.0])
        node = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["Y"])
        node.attribute.extend([helper.make_attribute("kernel_shape", [3])])
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 1, 3])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1, 1])],
            [W_tensor, B_tensor],
        )
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup(
            {f"x{i}": table[f"x{i}"] for i in range(3)}
        )

        ConvTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y")
        assert len(result) == 1  # 1 filter  1 output position
        backend = ibis.duckdb.connect()
        val = backend.execute(list(result.values())[0]).tolist()[0]
        # kernel picks x0=1.0, bias=10  11.0
        assert abs(val - 11.0) < 1e-6

    def test_conv1d_multi_filter(self):
        """Two filters, single channel: each filter is a different scalar."""
        from orbital.translation.steps.conv import ConvTranslator

        table = ibis.memtable({"x0": [2.0]})

        # W shape [2, 1, 1]: filter0=3.0, filter1=-1.0
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [2, 1, 1], [3.0, -1.0])
        B_tensor = helper.make_tensor("B", TensorProto.FLOAT, [2], [0.0, 0.0])
        node = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["Y"])
        node.attribute.extend([helper.make_attribute("kernel_shape", [1])])
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 1, 1])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 2, 1])],
            [W_tensor, B_tensor],
        )
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"]})

        ConvTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y")
        assert len(result) == 2  # 2 filters  1 position
        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        assert abs(vals[0] - 6.0) < 1e-6  # 2.0 * 3.0
        assert abs(vals[1] - (-2.0)) < 1e-6  # 2.0 * -1.0

    def test_conv1d_rejects_2d_weights(self):
        """Conv with 4-D weight tensor raises NotImplementedError."""
        from orbital.translation.steps.conv import ConvTranslator

        table = ibis.memtable({"x0": [1.0]})
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, 1, 1, 1], [1.0])
        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"])
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])],
            [W_tensor],
        )
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"]})

        with pytest.raises(NotImplementedError, match="1-D convolutions"):
            ConvTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()

    def test_conv_depthwise_group_equals_c_in(self):
        """Depthwise conv (group=C_in=2, C_out=2, k=1): each channel independently scaled."""
        # 2 channels, 1 position each; kernel [1]: W[0,0,0]=2, W[1,0,0]=3 (scale each channel)
        table = ibis.memtable({"c0": [4.0], "c1": [5.0]})
        W_flat = [2.0, 3.0]  # W[0,0,0]=2, W[1,0,0]=3
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [2, 1, 1], W_flat)
        node = helper.make_node(
            "Conv",
            inputs=["X", "W"],
            outputs=["Y"],
            group=2,
            dilations=[1],
            strides=[1],
            pads=[0, 0],
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])],
            [W_tensor],
        )
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"c0": table["c0"], "c1": table["c1"]})
        from orbital.translation.steps.conv import ConvTranslator

        ConvTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("Y")
        backend = ibis.duckdb.connect()
        assert abs(list(backend.execute(result["out_0_0"]))[0] - 8.0) < 1e-6  # 4*2
        assert abs(list(backend.execute(result["out_1_0"]))[0] - 15.0) < 1e-6  # 5*3

    def test_conv_grouped_general(self):
        """group=2 with C_in=4, C_out=2 (general grouped conv) works correctly."""
        # Input: 4 channels, 1 position — [1.0, 2.0, 3.0, 4.0]
        # Weight: shape [2, 2, 1], all 1.0 → each output is sum of its input group
        # Group 0: out_0_0 = c0+c1 = 1+2 = 3; Group 1: out_1_0 = c2+c3 = 3+4 = 7
        table = ibis.memtable({"c0": [1.0], "c1": [2.0], "c2": [3.0], "c3": [4.0]})
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [2, 2, 1], [1.0] * 4)
        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=2)
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])],
            [W_tensor],
        )
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup(
            {"c0": table["c0"], "c1": table["c1"], "c2": table["c2"], "c3": table["c3"]}
        )
        from orbital.translation.steps.conv import ConvTranslator

        translator = ConvTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("Y")
        assert isinstance(result, VariablesGroup)
        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        assert abs(vals[0] - 3.0) < 1e-5
        assert abs(vals[1] - 7.0) < 1e-5


# ---------------------------------------------------------------------------
# Unit tests: LSTMTranslator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit tests: LSTMTranslator
# ---------------------------------------------------------------------------
