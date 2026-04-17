"""Tests for ConvTranspose and pooling pipeline step translators."""

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


class TestConvTransposeTranslator:
    """Tests for the ONNX ConvTranspose (1-D transposed convolution) translator."""

    def test_convtranspose_registered(self):
        from orbital.translation.steps.convtranspose import ConvTransposeTranslator

        assert TRANSLATORS.get("ConvTranspose") is ConvTransposeTranslator

    def test_convtranspose_stride1_no_pad(self):
        """ConvTranspose with stride=1, no padding: output wider by (kW-1).

        Input  : [1.0, 2.0]  shape [1, 1, 2] (C_in=1, W_in=2)
        Kernel : [1, 1, 1]  shape [1, 1, 3]  (C_in=1, C_out=1, kW=3)
        Output width: (2-1)*1 - 0 - 0 + 3 = 4
          o=0: kernel k=0 hits i=0 → 1*1 = 1
          o=1: k=0→i=1 (2*1=2), k=1→i=0 (1*1=1) → 3
          o=2: k=1→i=1 (2*1=2), k=2→i=0 (1*1=1) → 3
          o=3: k=2→i=1 (2*1=2) → 2
        """
        from orbital.translation.steps.convtranspose import ConvTransposeTranslator

        table = ibis.memtable({"x0": [1.0], "x1": [2.0]})
        W_tensor = helper.make_tensor(
            "W", TensorProto.FLOAT, [1, 1, 3], [1.0, 1.0, 1.0]
        )
        node = helper.make_node("ConvTranspose", inputs=["X", "W"], outputs=["Y"])
        node.attribute.extend([helper.make_attribute("kernel_shape", [3])])
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1, 4])],
            [W_tensor],
        )
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"], "x1": table["x1"]})

        ConvTransposeTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 4  # C_out=1, W_out=4
        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        assert vals == [1.0, 3.0, 3.0, 2.0]

    def test_convtranspose_stride2_upsampling(self):
        """ConvTranspose stride=2 doubles spatial resolution (nearest-neighbour style).

        Input  : [3.0, 5.0]  C_in=1, W_in=2
        Kernel : [1.0]  shape [1, 1, 1]  (C_in=1, C_out=1, kW=1) — identity
        W_out: (2-1)*2 + 1 = 3
          o=0: i=(0-0)/2=0 valid (k=0) → 3
          o=1: i=(1-0)/2=0.5 → not integer → 0
          o=2: i=(2-0)/2=1 valid (k=0) → 5
        """
        from orbital.translation.steps.convtranspose import ConvTransposeTranslator

        table = ibis.memtable({"x0": [3.0], "x1": [5.0]})
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, 1, 1], [1.0])
        node = helper.make_node(
            "ConvTranspose",
            inputs=["X", "W"],
            outputs=["Y"],
            strides=[2],
            kernel_shape=[1],
        )
        node.attribute.extend(
            [
                helper.make_attribute("strides", [2]),
                helper.make_attribute("kernel_shape", [1]),
            ]
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1, 3])],
            [W_tensor],
        )
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"], "x1": table["x1"]})

        ConvTransposeTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 3  # W_out = (2-1)*2 + 1 = 3
        backend = ibis.duckdb.connect()

        def _exec_val(v):
            r = backend.execute(v)
            return r.tolist()[0] if hasattr(r, "tolist") else float(r)

        vals = [_exec_val(v) for v in result.values()]
        assert abs(vals[0] - 3.0) < 1e-6
        assert abs(vals[1] - 0.0) < 1e-6
        assert abs(vals[2] - 5.0) < 1e-6

    def test_convtranspose_with_bias(self):
        """ConvTranspose adds per-output-channel bias to every output position."""
        from orbital.translation.steps.convtranspose import ConvTransposeTranslator

        table = ibis.memtable({"x0": [2.0]})
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, 1, 1], [1.0])
        B_tensor = helper.make_tensor("B", TensorProto.FLOAT, [1], [10.0])
        node = helper.make_node("ConvTranspose", inputs=["X", "W", "B"], outputs=["Y"])
        node.attribute.extend([helper.make_attribute("kernel_shape", [1])])
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 1, 1])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1, 1])],
            [W_tensor, B_tensor],
        )
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"]})

        ConvTransposeTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y")
        backend = ibis.duckdb.connect()
        val = backend.execute(list(result.values())[0]).tolist()[0]
        assert abs(val - 12.0) < 1e-6  # 2*1 + 10.0 = 12

    def test_convtranspose_multi_channel_out(self):
        """ConvTranspose expands C_in=1 to C_out=2 per output position.

        W shape [1, 2, 1]: C_in=1, C_out=2, kW=1
          W[0, 0, 0]=3.0, W[0, 1, 0]=-1.0
        Input: [2.0]  (C_in=1, W_in=1)
        Output: W_out=1, C_out=2
          Y[0, 0] = 2*3 = 6
          Y[1, 0] = 2*(-1) = -2
        """
        from orbital.translation.steps.convtranspose import ConvTransposeTranslator

        table = ibis.memtable({"x0": [2.0]})
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, 2, 1], [3.0, -1.0])
        node = helper.make_node("ConvTranspose", inputs=["X", "W"], outputs=["Y"])
        node.attribute.extend([helper.make_attribute("kernel_shape", [1])])
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 1, 1])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 2, 1])],
            [W_tensor],
        )
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"]})

        ConvTransposeTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 2  # C_out=2, W_out=1
        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        assert abs(vals[0] - 6.0) < 1e-6
        assert abs(vals[1] - (-2.0)) < 1e-6

    def test_convtranspose_group_nonone_raises(self):
        """ConvTranspose with unsupported group (not 1 and not C_in) raises NotImplementedError."""
        from orbital.translation.steps.convtranspose import ConvTransposeTranslator

        # group=2 but C_in=1 → group != 1 and group != c_in → must raise
        table = ibis.memtable({"x0": [1.0]})
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, 1, 1], [1.0])
        node = helper.make_node("ConvTranspose", inputs=["X", "W"], outputs=["Y"])
        node.attribute.extend(
            [
                helper.make_attribute("kernel_shape", [1]),
                helper.make_attribute("group", 2),
            ]
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 1, 1])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1, 1])],
            [W_tensor],
        )
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"]})
        with pytest.raises(NotImplementedError):
            ConvTransposeTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()

    def test_convtranspose_depthwise(self):
        """ConvTranspose with group=C_in (depthwise): each input ch has its own filter.

        C_in=2, C_out_per_group=1, kW=1, group=2 → C_out=2.
        Weight W[ci, 0, 0] = ci+1: W[0,0,0]=1, W[1,0,0]=2.
        Output Y[co, 0] = X[co, 0] * W[co, 0, 0].
        With X=[3.0, 5.0]: Y=[3.0, 10.0].
        """
        from orbital.translation.steps.convtranspose import ConvTransposeTranslator

        # Weight shape (C_in=2, C_out_per_group=1, kW=1)
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [2, 1, 1], [1.0, 2.0])
        node = helper.make_node("ConvTranspose", inputs=["X", "W"], outputs=["Y"])
        node.attribute.extend(
            [
                helper.make_attribute("kernel_shape", [1]),
                helper.make_attribute("group", 2),
            ]
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 2])],
            [W_tensor],
        )
        table = ibis.memtable({"x0": [3.0], "x1": [5.0]})
        variables = GraphVariables(ibis.memtable({"X": [0.0]}), graph)
        variables["X"] = ValueVariablesGroup({"x0": table["x0"], "x1": table["x1"]})

        ConvTransposeTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()

        result = variables.peek_variable("Y")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 2

        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        assert vals == pytest.approx([3.0, 10.0])
