"""Tests for matrix pipeline step translators."""

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

class TestArgMaxTranslator:

    def test_argmax_group_columns(self):
        """Test ArgMaxTranslator with a group of columns."""
        table = ibis.memtable(
            {
                "class_0": [1.0, 5.0, 2.0],
                "class_1": [3.0, 2.0, 8.0],
                "class_2": [2.0, 1.0, 3.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMax <axis: int = 1, keepdims: int = 1> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "class_0": table["class_0"],
                "class_1": table["class_1"],
                "class_2": table["class_2"],
            }
        )

        translator = ArgMaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        # Row 0: max is class_1 (3.0) -> index 1
        # Row 1: max is class_0 (5.0) -> index 0
        # Row 2: max is class_1 (8.0) -> index 1
        assert computed == [1, 0, 1]

    def test_argmax_single_column(self):
        """Test ArgMaxTranslator raises error for single column input."""
        table = ibis.memtable({"data": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMax <axis: int = 1> (data)
            }
        """)

        variables = GraphVariables(table, model)

        translator = ArgMaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(NotImplementedError, match="can only be applied to a group"):
            translator.process()

    def test_argmax_unsupported_axis(self):
        """Test ArgMaxTranslator raises error for axis != 1."""
        table = ibis.memtable({"data": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMax <axis: int = 0, keepdims: int = 1> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "class_0": table["data"],
                "class_1": table["data"],
            }
        )

        translator = ArgMaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(NotImplementedError, match="only supports axis=1"):
            translator.process()

    def test_argmax_unsupported_keepdims(self):
        """Test ArgMaxTranslator raises error for keepdims != 1."""
        table = ibis.memtable({"data": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMax <axis: int = 1, keepdims: int = 0> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "class_0": table["data"],
                "class_1": table["data"],
            }
        )

        translator = ArgMaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError, match="only supports retaining original"
        ):
            translator.process()

    def test_argmax_empty_group(self):
        """Test ArgMaxTranslator raises error for empty group."""
        table = ibis.memtable({"data": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMax <axis: int = 1, keepdims: int = 1> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup({})

        translator = ArgMaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="requires at least one column"):
            translator.process()

    def test_argmax_single_key_in_group(self):
        """Test ArgMaxTranslator with single key returns that value."""
        table = ibis.memtable({"class_0": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMax <axis: int = 1, keepdims: int = 1> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "class_0": table["class_0"],
            }
        )

        translator = ArgMaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables

    def test_argmax_select_last_index(self):
        """Test ArgMaxTranslator with select_last_index=1."""
        table = ibis.memtable(
            {
                "class_0": [3.0, 3.0],
                "class_1": [3.0, 3.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMax <axis: int = 1, keepdims: int = 1, select_last_index: int = 1> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "class_0": table["class_0"],
                "class_1": table["class_1"],
            }
        )

        translator = ArgMaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        # Just verify the translator processes without error and exercises the code path
        assert "output" in variables



class TestReshapeTranslator:

    def test_reshape_single_column_to_single_column(self):
        """Test ReshapeTranslator with shape=[-1] on single column (passes through)."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <int64[1] shape = {-1}>
            {
                output = Reshape(input, shape)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ReshapeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Verify output equals input (passthrough)
        backend = ibis.duckdb.connect()
        input_values = list(backend.execute(table["input"]))
        output_values = list(backend.execute(result))
        assert output_values == input_values

    def test_reshape_group_to_same_size(self):
        """Test ReshapeTranslator with shape=[-1, N] on N columns (passes through)."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
                "col_c": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <int64[2] shape = {-1, 3}>
            {
                output = Reshape(input, shape)
            }
        """)

        # Use dummy table for GraphVariables since we override the input
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
                "col_c": table["col_c"],
            }
        )

        translator = ReshapeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Should return the same NumericVariablesGroup (passthrough)
        assert isinstance(result, NumericVariablesGroup)
        assert len(result) == 3

        # Verify all columns preserved with same values
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["col_b"])) == [4.0, 5.0, 6.0]
        assert list(backend.execute(result["col_c"])) == [7.0, 8.0, 9.0]

    def test_reshape_requires_integer_shape(self):
        """Test ReshapeTranslator raises error when shape is not integers."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] shape = {-1.0}>
            {
                output = Reshape(input, shape)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ReshapeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError, match="requires integer values for the shape"
        ):
            translator.process()

    def test_reshape_cannot_change_row_count(self):
        """Test ReshapeTranslator raises error when shape[0] != -1."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <int64[2] shape = {3, 1}>
            {
                output = Reshape(input, shape)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ReshapeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError, match="Reshape can't change the number of rows"
        ):
            translator.process()

    def test_reshape_unsupported_shape(self):
        """Test ReshapeTranslator raises error for unsupported shape combinations."""
        # Test case: shape=[-1, 2] but input has 3 columns
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
                "col_c": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <int64[2] shape = {-1, 2}>
            {
                output = Reshape(input, shape)
            }
        """)

        # Use dummy table for GraphVariables since we override the input
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
                "col_c": table["col_c"],
            }
        )

        translator = ReshapeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(NotImplementedError, match="Reshape"):
            translator.process()



class TestTransposeTranslator:
    """Tests for TransposeTranslator."""


    def test_transpose_registered(self):
        """Verify TransposeTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.transpose import TransposeTranslator
        assert TRANSLATORS.get("Transpose") is TransposeTranslator

    def test_transpose_passthrough(self):
        """Transpose is a pass-through for column groups."""
        table = ibis.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Transpose <perm: ints = [1, 0]> (input)
            }
        """)
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.transpose import TransposeTranslator
        t = TransposeTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert set(result.keys()) == {"a", "b"}



class TestFlattenTranslator:
    """Tests for FlattenTranslator."""


    def test_flatten_registered(self):
        """Verify FlattenTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.flatten import FlattenTranslator
        assert TRANSLATORS.get("Flatten") is FlattenTranslator

    def test_flatten_axis1_passthrough(self):
        """Flatten at axis=1 is a pass-through (the only supported case)."""
        table = ibis.memtable({"a": [1.0], "b": [2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Flatten <axis: int = 1> (input)
            }
        """)
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.flatten import FlattenTranslator
        t = FlattenTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert set(result.keys()) == {"a", "b"}

    def test_flatten_non_axis1_raises(self):
        """Flatten at axis != 1 raises NotImplementedError."""
        table = ibis.memtable({"x": [1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Flatten <axis: int = 0> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.flatten import FlattenTranslator
        t = FlattenTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        with pytest.raises(NotImplementedError, match="axis=1"):
            t.process()



class TestSqueezeTranslator:
    """Tests for SqueezeTranslator."""


    def test_squeeze_registered(self):
        from orbital.translation.steps.squeeze import SqueezeTranslator
        assert TRANSLATORS.get("Squeeze") is SqueezeTranslator

    def test_squeeze_passthrough(self):
        """Squeeze is a pass-through in columnar representation."""
        table = ibis.memtable({"x": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Squeeze(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.squeeze import SqueezeTranslator
        t = SqueezeTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [1.0, 2.0, 3.0]



class TestUnsqueezeTranslator:
    """Tests for UnsqueezeTranslator."""


    def test_unsqueeze_registered(self):
        from orbital.translation.steps.squeeze import UnsqueezeTranslator
        assert TRANSLATORS.get("Unsqueeze") is UnsqueezeTranslator

    def test_unsqueeze_passthrough(self):
        """Unsqueeze is a pass-through in columnar representation."""
        table = ibis.memtable({"x": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <int64[1] axes = {0}>
            {
                output = Unsqueeze(x, axes)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.squeeze import UnsqueezeTranslator
        t = UnsqueezeTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [1.0, 2.0, 3.0]



class TestShapeTranslator:
    """Tests for ShapeTranslator (stub — always raises)."""


    def test_shape_registered(self):
        from orbital.translation.steps.shape import ShapeTranslator
        assert TRANSLATORS.get("Shape") is ShapeTranslator

    def test_shape_raises(self):
        """Shape must always raise NotImplementedError."""
        node = helper.make_node("Shape", inputs=["x"], outputs=["y"])
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("y", TensorProto.INT64, [None])],
            [],
        )
        table = ibis.memtable({"x": [1.0]})
        variables = GraphVariables(table, graph)
        variables["x"] = table["x"]
        from orbital.translation.steps.shape import ShapeTranslator
        t = ShapeTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        with pytest.raises(NotImplementedError):
            t.process()


# ---------------------------------------------------------------------------
# Unit tests: TileTranslator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit tests: TileTranslator
# ---------------------------------------------------------------------------


