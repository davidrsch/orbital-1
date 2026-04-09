"""Tests for unary elementwise pipeline step translators."""

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


class TestNegTranslator:
    """Tests for NegTranslator."""


    def test_neg_registered(self):
        from orbital.translation.steps.neg import NegTranslator
        assert TRANSLATORS.get("Neg") is NegTranslator

    def test_neg_single_column(self):
        table = ibis.memtable({"x": [1.0, -2.0, 0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Neg(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.neg import NegTranslator
        t = NegTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [-1.0, 2.0, 0.0]

    def test_neg_group(self):
        table = ibis.memtable({"a": [1.0, -2.0], "b": [-3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Neg(x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.neg import NegTranslator
        t = NegTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["a"])) == [-1.0, 2.0]
        assert list(backend.execute(result["b"])) == [3.0, -4.0]



class TestPowTranslator:
    """Tests for PowTranslator."""


    def test_pow_registered(self):
        from orbital.translation.steps.pow import PowTranslator
        assert TRANSLATORS.get("Pow") is PowTranslator

    def test_pow_single_column_constant_exp(self):
        """x^3 with constant exponent from initializer."""
        table = ibis.memtable({"x": [2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <float[1] exp = {3.0}>
            {
                output = Pow(x, exp)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.pow import PowTranslator
        t = PowTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [8.0, 27.0]

    def test_pow_group_constant_exp(self):
        """Group raised to constant exponent."""
        table = ibis.memtable({"a": [2.0, 3.0], "b": [4.0, 5.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <float[1] exp = {2.0}>
            {
                output = Pow(x, exp)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.pow import PowTranslator
        t = PowTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["a"])) == [4.0, 9.0]
        assert list(backend.execute(result["b"])) == [16.0, 25.0]


# ---------------------------------------------------------------------------
# Unit tests: ExpTranslator
# ---------------------------------------------------------------------------



class TestModTranslator:
    """Tests for ModTranslator."""


    def test_mod_registered(self):
        from orbital.translation.steps.mod import ModTranslator
        assert TRANSLATORS.get("Mod") is ModTranslator

    def test_mod_positive(self):
        """7 % 3 = 1."""
        divisor_tensor = helper.make_tensor("d", TensorProto.FLOAT, [1], [3.0])
        node = helper.make_node("Mod", inputs=["x", "d"], outputs=["y"], fmod=0)
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            [divisor_tensor],
        )
        table = ibis.memtable({"x": [7.0]})
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = table["x"]
        from orbital.translation.steps.mod import ModTranslator
        ModTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("y")
        backend = ibis.duckdb.connect()
        assert abs(list(backend.execute(result))[0] - 1.0) < 1e-6

    def test_fmod_negative(self):
        """fmod(-7, 3): remainder has sign of dividend → -1."""
        divisor_tensor = helper.make_tensor("d", TensorProto.FLOAT, [1], [3.0])
        node = helper.make_node("Mod", inputs=["x", "d"], outputs=["y"], fmod=1)
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            [divisor_tensor],
        )
        table = ibis.memtable({"x": [-7.0]})
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = table["x"]
        from orbital.translation.steps.mod import ModTranslator
        ModTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("y")
        backend = ibis.duckdb.connect()
        val = list(backend.execute(result))[0]
        assert abs(val - (-1.0)) < 1e-6


# ---------------------------------------------------------------------------
# Unit tests: ScatterElementsTranslator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit tests: PadTranslator
# ---------------------------------------------------------------------------



class TestPadTranslator:
    """Tests for PadTranslator (constant padding on feature axis)."""


    def test_pad_registered(self):
        from orbital.translation.steps.pad import PadTranslator
        assert TRANSLATORS.get("Pad") is PadTranslator

    def test_pad_rank1_adds_zero_columns(self):
        """pads=[1, 2] on a rank-1 vector [3.0, 4.0]: [0, 3, 4, 0, 0]."""
        table = ibis.memtable({"a": [3.0], "b": [4.0]})
        pads_tensor = helper.make_tensor("pads", TensorProto.INT64, [2], [1, 2])
        node = helper.make_node("Pad", inputs=["x", "pads"], outputs=["y"])
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            [pads_tensor],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.pad import PadTranslator
        PadTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("y")
        backend = ibis.duckdb.connect()
        # 1 zero before + 2 original columns + 2 zeros after = 5 columns total
        assert len(result) == 5
        assert float(backend.execute(result["pad_begin_0"])) == 0.0
        assert list(backend.execute(result["data_0"]))[0] == 3.0
        assert list(backend.execute(result["data_1"]))[0] == 4.0
        assert float(backend.execute(result["pad_end_0"])) == 0.0
        assert float(backend.execute(result["pad_end_1"])) == 0.0

    def test_pad_rank2_batch_dim_must_be_zero(self):
        """pads=[1, 0, 0, 0] pads the batch dim — must raise NotImplementedError."""
        table = ibis.memtable({"a": [1.0]})
        pads_tensor = helper.make_tensor("pads", TensorProto.INT64, [4], [1, 0, 0, 0])
        node = helper.make_node("Pad", inputs=["x", "pads"], outputs=["y"])
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            [pads_tensor],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = table["a"]
        from orbital.translation.steps.pad import PadTranslator
        with pytest.raises(NotImplementedError, match="batch"):
            PadTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()

    def test_pad_mode_reflect_raises(self):
        """mode='reflect' should raise NotImplementedError."""
        table = ibis.memtable({"a": [1.0]})
        pads_tensor = helper.make_tensor("pads", TensorProto.INT64, [2], [1, 1])
        node = helper.make_node(
            "Pad", inputs=["x", "pads"], outputs=["y"], mode="reflect"
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            [pads_tensor],
        )
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), graph)
        variables["x"] = table["a"]
        from orbital.translation.steps.pad import PadTranslator
        with pytest.raises(NotImplementedError, match="reflect"):
            PadTranslator(
                table, graph.node[0], variables, self.optimizer, TranslationOptions()
            ).process()


