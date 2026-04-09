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
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, 1, 3], [1.0, 1.0, 1.0])
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
            "ConvTranspose", inputs=["X", "W"], outputs=["Y"],
            strides=[2], kernel_shape=[1],
        )
        node.attribute.extend([
            helper.make_attribute("strides", [2]),
            helper.make_attribute("kernel_shape", [1]),
        ])
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
        """ConvTranspose with group > 1 raises NotImplementedError."""
        from orbital.translation.steps.convtranspose import ConvTransposeTranslator

        table = ibis.memtable({"x0": [1.0]})
        W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, 1, 1], [1.0])
        node = helper.make_node("ConvTranspose", inputs=["X", "W"], outputs=["Y"])
        node.attribute.extend([
            helper.make_attribute("kernel_shape", [1]),
            helper.make_attribute("group", 2),
        ])
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


# ---------------------------------------------------------------------------
# Unit tests: LSTMTranslator
# ---------------------------------------------------------------------------


class TestGlobalAveragePoolTranslator:

    def test_globalavgpool_registered(self):
        from orbital.translation.steps.globalavgpool import GlobalAveragePoolTranslator
        assert TRANSLATORS.get("GlobalAveragePool") is GlobalAveragePoolTranslator

    def test_globalavgpool_correctness(self):
        """GlobalAveragePool reduces a group to the per-row mean."""
        from orbital.translation.steps.globalavgpool import GlobalAveragePoolTranslator
        table = ibis.memtable({"c0": [1.0, 4.0], "c1": [3.0, 8.0], "c2": [2.0, 6.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = GlobalAveragePool(input)
            }
        """)
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {"c0": table["c0"], "c1": table["c1"], "c2": table["c2"]}
        )
        GlobalAveragePoolTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = ibis.duckdb.connect().execute(
            variables.peek_variable("output")
        ).tolist()
        assert result == [2.0, 6.0]  # (1+3+2)/3=2, (4+8+6)/3=6


class TestGlobalMaxPoolTranslator:

    def test_globalmaxpool_registered(self):
        from orbital.translation.steps.globalmaxpool import GlobalMaxPoolTranslator
        assert TRANSLATORS.get("GlobalMaxPool") is GlobalMaxPoolTranslator

    def test_globalmaxpool_correctness(self):
        """GlobalMaxPool reduces a group to the per-row maximum."""
        from orbital.translation.steps.globalmaxpool import GlobalMaxPoolTranslator
        table = ibis.memtable({"c0": [1.0, 9.0], "c1": [5.0, 2.0], "c2": [3.0, 7.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = GlobalMaxPool(input)
            }
        """)
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {"c0": table["c0"], "c1": table["c1"], "c2": table["c2"]}
        )
        GlobalMaxPoolTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = ibis.duckdb.connect().execute(
            variables.peek_variable("output")
        ).tolist()
        assert result == [5.0, 9.0]  # max(1,5,3)=5, max(9,2,7)=9


class TestAveragePoolTranslator:
    """Tests for AveragePoolTranslator."""


    def test_averagepool_registered(self):
        from orbital.translation.steps.avgpool import AveragePoolTranslator
        assert TRANSLATORS.get("AveragePool") is AveragePoolTranslator

    def test_averagepool_kernel1_passthrough(self):
        """AveragePool with kernel=1 is a pass-through."""
        table = ibis.memtable({"x": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = AveragePool <kernel_shape: ints = [1]> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.avgpool import AveragePoolTranslator
        t = AveragePoolTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [1.0, 2.0, 3.0]

    def test_averagepool_global(self):
        """AveragePool with kernel covering all columns returns mean."""
        table = ibis.memtable({"a": [2.0, 6.0], "b": [4.0, 10.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = AveragePool <kernel_shape: ints = [2]> (x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.avgpool import AveragePoolTranslator
        t = AveragePoolTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        out = variables.peek_variable("output")
        from orbital.translation.variables import ValueVariablesGroup
        assert isinstance(out, ValueVariablesGroup)
        result = list(backend.execute(list(out.values())[0]))
        assert result == [3.0, 8.0]

    def test_averagepool_sliding_window(self):
        """AveragePool sliding window: kernel=2, stride=1, L_in=4 → L_out=3."""
        from orbital.translation.steps.avgpool import AveragePoolTranslator
        from orbital.translation.variables import ValueVariablesGroup
        table = ibis.memtable({"c0": [1.0], "c1": [2.0], "c2": [3.0], "c3": [4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = AveragePool <kernel_shape: ints = [2], strides: ints = [1]> (x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup(
            {"c0": table["c0"], "c1": table["c1"], "c2": table["c2"], "c3": table["c3"]}
        )
        AveragePoolTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        out = variables.peek_variable("output")
        assert isinstance(out, ValueVariablesGroup)
        assert len(out) == 3  # L_out = (4-2)//1 + 1 = 3
        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in out.values()]
        assert vals == [1.5, 2.5, 3.5]  # (1+2)/2, (2+3)/2, (3+4)/2

    def test_averagepool_multidim_raises(self):
        """AveragePool with 2-D kernel_shape raises NotImplementedError."""
        table = ibis.memtable({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = AveragePool <kernel_shape: ints = [2, 2]> (x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup(
            {"a": table["a"], "b": table["b"], "c": table["c"], "d": table["d"]}
        )
        from orbital.translation.steps.avgpool import AveragePoolTranslator
        t = AveragePoolTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        with pytest.raises(NotImplementedError):
            t.process()


class TestMaxPoolTranslator:

    def test_maxpool_registered(self):
        from orbital.translation.steps.maxpool import MaxPoolTranslator
        assert TRANSLATORS.get("MaxPool") is MaxPoolTranslator

    def test_maxpool_global_correctness(self):
        """MaxPool with kernel_shape=[n_cols] returns the per-row maximum."""
        from orbital.translation.steps.maxpool import MaxPoolTranslator
        from orbital.translation.variables import ValueVariablesGroup
        table = ibis.memtable({"c0": [1.0, 9.0], "c1": [5.0, 2.0], "c2": [3.0, 7.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = MaxPool <kernel_shape: ints = [3]> (input)
            }
        """)
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {"c0": table["c0"], "c1": table["c1"], "c2": table["c2"]}
        )
        MaxPoolTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        out = variables.peek_variable("output")
        assert isinstance(out, ValueVariablesGroup)
        result = ibis.duckdb.connect().execute(
            list(out.values())[0]
        ).tolist()
        assert result == [5.0, 9.0]  # max(1,5,3)=5, max(9,2,7)=9

    def test_maxpool_sliding_window(self):
        """MaxPool sliding window: kernel=2, stride=1, L_in=4 → L_out=3."""
        from orbital.translation.steps.maxpool import MaxPoolTranslator
        from orbital.translation.variables import ValueVariablesGroup
        table = ibis.memtable({"c0": [1.0], "c1": [4.0], "c2": [2.0], "c3": [5.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = MaxPool <kernel_shape: ints = [2], strides: ints = [1]> (input)
            }
        """)
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {"c0": table["c0"], "c1": table["c1"], "c2": table["c2"], "c3": table["c3"]}
        )
        MaxPoolTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        out = variables.peek_variable("output")
        assert isinstance(out, ValueVariablesGroup)
        assert len(out) == 3  # L_out = (4-2)//1 + 1 = 3
        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in out.values()]
        assert vals == [4.0, 4.0, 5.0]  # max(1,4), max(4,2), max(2,5)


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
# Unit tests: ConvTranslator
# ---------------------------------------------------------------------------


class TestDropoutTranslator:

    def test_dropout_registered(self):
        from orbital.translation.steps.dropout import DropoutTranslator
        assert TRANSLATORS.get("Dropout") is DropoutTranslator

    def test_dropout_is_identity_at_inference(self):
        """Dropout is a pass-through at inference time (ratio is ignored)."""
        from orbital.translation.steps.dropout import DropoutTranslator
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Dropout(input)
            }
        """)
        variables = GraphVariables(table, model)
        DropoutTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = ibis.duckdb.connect().execute(
            variables.peek_variable("output")
        ).tolist()
        assert result == [1.0, 2.0, 3.0]


