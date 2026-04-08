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

class TestMatMulTranslator:

    def test_matmul_single_column_times_1d_weight_vector_length_1(self):
        """Test MatMulTranslator with single column times 1D weight vector (coefficient vector of length 1)."""
        table = ibis.memtable({"input": [2.0, 3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] weights = {5.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        variables = GraphVariables(table, model)
        translator = MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [10.0, 15.0, 20.0]

    def test_matmul_group_columns_times_2d_weight_matrix_multiple_outputs(self):
        """Test MatMulTranslator with group of columns times 2D weight matrix (produces multiple outputs)."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
            }
        )
        # Flattened row-major: [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[6] weights = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        model.initializer[0].dims[:] = [2, 3]

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 3

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["out_0"])) == [8.5, 11.0, 13.5]
        assert list(backend.execute(result["out_1"])) == [11.0, 14.5, 18.0]
        assert list(backend.execute(result["out_2"])) == [13.5, 18.0, 22.5]

    def test_matmul_group_columns_times_1d_weight_vector_single_output(self):
        """Test MatMulTranslator with group of columns times 1D weight vector (produces single output)."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
                "col_c": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[3] weights = {2.0, 3.0, 4.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        model.initializer[0].dims[:] = [3]

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
                "col_c": table["col_c"],
            }
        )

        translator = MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [42.0, 51.0, 60.0]

    def test_matmul_with_optimizer_enabled(self):
        """Test MatMulTranslator with optimizer folding enabled."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] weights = {2.0, 3.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        model.initializer[0].dims[:] = [2]

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = MatMulTranslator(
            table,
            model.node[0],
            variables,
            Optimizer(enabled=True),
            TranslationOptions(),
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [14.0, 19.0, 24.0]

    def test_matmul_error_non_numeric_input_columns(self):
        """Test MatMulTranslator raises error for non-numeric input columns."""
        table = ibis.memtable({"input": ["a", "b", "c"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] input) => (float[N] output)
            <float[1] weights = {2.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        model.initializer[0].dims[:] = [1]
        variables = GraphVariables(table, model)

        translator = MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="first operand must be a numeric column"):
            translator.process()

    def test_matmul_error_weight_matrix_dimension_mismatch(self):
        """Test MatMulTranslator raises error for weight matrix dimension mismatch."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
            }
        )
        # Weight matrix [3, 2] expects 3 features but input provides only 2
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[6] weights = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        model.initializer[0].dims[:] = [3, 2]

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="Mismatch: number of features"):
            translator.process()

    def test_matmul_error_unsupported_weight_tensor_rank(self):
        """Test MatMulTranslator raises error for unsupported weight tensor shapes (rank > 2)."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[8] weights = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        model.initializer[0].dims[:] = [2, 2, 2]  # 3D tensor - unsupported

        variables = GraphVariables(table, model)

        translator = MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError, match="coefficient tensor rank > 2 is not supported"
        ):
            translator.process()

    def test_matmul_error_single_column_with_coefficient_vector_longer_than_1(self):
        """Test MatMulTranslator raises error for single column with coefficient vector longer than 1."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[3] weights = {2.0, 3.0, 4.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        # Set weight vector dimensions [3] - but we have single column
        model.initializer[0].dims[:] = [3]

        variables = GraphVariables(table, model)

        translator = MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="Expected coefficient vector of length 1 for single operand",
        ):
            translator.process()

    def test_matmul_group_columns_times_2d_weight_matrix_single_output(self):
        """Test MatMulTranslator with group of columns times 2D weight matrix with single output."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] weights = {2.0, 3.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        model.initializer[0].dims[:] = [2, 1]

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert not isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [14.0, 19.0, 24.0]

    def test_matmul_single_column_times_2d_weight_matrix_multiple_outputs(self):
        """Test MatMulTranslator with single column times 2D weight matrix [1, N] (produces multiple outputs)."""
        table = ibis.memtable({"input": [2.0, 3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[3] weights = {2.0, 3.0, 4.0}>
            {
                output = MatMul(input, weights)
            }
        """)

        model.initializer[0].dims[:] = [1, 3]

        variables = GraphVariables(table, model)

        translator = MatMulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 3

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["out_0"])) == [4.0, 6.0, 8.0]
        assert list(backend.execute(result["out_1"])) == [6.0, 9.0, 12.0]
        assert list(backend.execute(result["out_2"])) == [8.0, 12.0, 16.0]



class TestGemmTranslator:
    """Tests for GemmTranslator — see test_mlp.py for comprehensive tests."""


    def test_gemm_registered(self):
        """Verify GemmTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.gemm import GemmTranslator
        assert TRANSLATORS.get("Gemm") is GemmTranslator

    def test_gemm_correctness(self):
        """Y = A @ B^T + C; verifies Gemm with transB=1."""
        from orbital.translation.steps.gemm import GemmTranslator
        table = ibis.memtable({"h0": [1.0, 2.0], "h1": [3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[4] B = {2.0, 0.0, 0.0, 3.0}, float[2] C = {1.0, 1.0}>
            {
                output = Gemm <transB: int = 1> (input, B, C)
            }
        """)
        model.initializer[0].dims[:] = [2, 2]  # B: (output_dim=2, input_dim=2)
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {"h0": table["h0"], "h1": table["h1"]}
        )
        GemmTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        backend = ibis.duckdb.connect()
        # Row 0: [1, 3] @ [[2,0],[0,3]]^T + [1,1] = [2+0+1, 0+9+1] = [3, 10]
        assert backend.execute(result["out_0"]).tolist() == [3.0, 5.0]
        assert backend.execute(result["out_1"]).tolist() == [10.0, 13.0]

    def test_gemm_no_bias(self):
        """Gemm without C input (2-operand form) must not crash.

        Regression test: previously raised IndexError when accessing self.inputs[2].
        """
        from orbital.translation.steps.gemm import GemmTranslator
        table = ibis.memtable({"h0": [1.0, 2.0], "h1": [3.0, 4.0]})
        # 2-input Gemm: Y = A @ B^T  (no C)
        B_tensor = helper.make_tensor("B", TensorProto.FLOAT, [2, 2], [2.0, 0.0, 0.0, 3.0])
        node = helper.make_node(
            "Gemm",
            inputs=["A", "B"],
            outputs=["output"],
            transB=1,
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
            [B_tensor],
        )
        variables = GraphVariables(ibis.memtable({"A": [1.0]}), graph)
        variables["A"] = NumericVariablesGroup({"h0": table["h0"], "h1": table["h1"]})
        GemmTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        backend = ibis.duckdb.connect()
        # Row 0: [1, 3] @ [[2,0],[0,3]]^T = [2, 9]; Row 1: [2, 4] @ ... = [4, 12]
        assert backend.execute(result["out_0"]).tolist() == [2.0, 4.0]
        assert backend.execute(result["out_1"]).tolist() == [9.0, 12.0]



class TestFeatureVectorizerTranslator:

    def test_featurevectorizer_multiple_inputs_with_dimensions(self):
        """Test vectorizing multiple inputs with inputdimensions attribute."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "group_col_a": [4.0, 5.0, 6.0],
                "group_col_b": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input1, float[N] input2) => (float[N] output) {
                output = ai.onnx.ml.FeatureVectorizer <inputdimensions: ints = [1, 2]> (input1, input2)
            }
        """)

        variables = GraphVariables(
            ibis.memtable({"input1": [1.0], "input2": [1.0]}), model
        )
        variables["input1"] = table["feature1"]
        variables["input2"] = ValueVariablesGroup(
            {
                "col_a": table["group_col_a"],
                "col_b": table["group_col_b"],
            }
        )

        translator = FeatureVectorizerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 3
        assert "input1" in result
        assert "input2.col_a" in result
        assert "input2.col_b" in result

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["input1"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["input2.col_a"])) == [4.0, 5.0, 6.0]
        assert list(backend.execute(result["input2.col_b"])) == [7.0, 8.0, 9.0]

    def test_featurevectorizer_single_input_feature(self):
        """Test single input feature."""
        table = ibis.memtable({"input1": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input1) => (float[N] output) {
                output = ai.onnx.ml.FeatureVectorizer <inputdimensions: ints = [1]> (input1)
            }
        """)

        variables = GraphVariables(table, model)
        translator = FeatureVectorizerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 1
        assert "input1" in result

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["input1"])) == [1.0, 2.0, 3.0]

    def test_featurevectorizer_mismatched_dimensions(self):
        """Test error when input dimensions don't match actual columns."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "group_col_a": [4.0, 5.0, 6.0],
                "group_col_b": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input1, float[N] input2) => (float[N] output) {
                output = ai.onnx.ml.FeatureVectorizer <inputdimensions: ints = [1, 3]> (input1, input2)
            }
        """)

        variables = GraphVariables(
            ibis.memtable({"input1": [1.0], "input2": [1.0]}), model
        )
        variables["input1"] = table["feature1"]
        variables["input2"] = ValueVariablesGroup(
            {
                "col_a": table["group_col_a"],
                "col_b": table["group_col_b"],
            }
        )

        translator = FeatureVectorizerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="Number of columns in input input2"):
            translator.process()

    def test_featurevectorizer_wrong_single_column_dimension(self):
        """Test error when single column has wrong dimension."""
        table = ibis.memtable({"input1": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input1) => (float[N] output) {
                output = ai.onnx.ml.FeatureVectorizer <inputdimensions: ints = [2]> (input1)
            }
        """)

        variables = GraphVariables(table, model)
        translator = FeatureVectorizerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="When merging over individual columns, the dimension should be 1",
        ):
            translator.process()

    def test_featurevectorizer_mismatched_input_count(self):
        """Test error when input count doesn't match dimensions count."""
        table = ibis.memtable({"input1": [1.0, 2.0, 3.0], "input2": [4.0, 5.0, 6.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input1, float[N] input2) => (float[N] output) {
                output = ai.onnx.ml.FeatureVectorizer <inputdimensions: ints = [1]> (input1, input2)
            }
        """)

        variables = GraphVariables(table, model)
        translator = FeatureVectorizerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="Number of input dimensions should be equal to number of inputs",
        ):
            translator.process()



class TestConcatTranslator:

    def test_concat_two_single_columns(self):
        """Test concatenating two single columns."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] feature1, float[N] feature2) => (float[N] output) {
                output = Concat <axis: int = 1> (feature1, feature2)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ConcatTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 2
        assert "feature1" in result
        assert "feature2" in result

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["feature1"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["feature2"])) == [4.0, 5.0, 6.0]

    def test_concat_column_groups(self):
        """Test concatenating column groups."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
                "col_c": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] group1, float[N] group2) => (float[N] output) {
                output = Concat <axis: int = -1> (group1, group2)
            }
        """)

        variables = GraphVariables(
            ibis.memtable({"group1": [1.0], "group2": [1.0]}), model
        )
        variables["group1"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )
        variables["group2"] = ValueVariablesGroup(
            {
                "col_c": table["col_c"],
            }
        )

        translator = ConcatTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 3
        assert "group1.col_a" in result
        assert "group1.col_b" in result
        assert "group2.col_c" in result

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["group1.col_a"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["group1.col_b"])) == [4.0, 5.0, 6.0]
        assert list(backend.execute(result["group2.col_c"])) == [7.0, 8.0, 9.0]

    def test_concat_mix_single_columns_and_groups(self):
        """Test concatenating mix of single columns and groups."""
        table = ibis.memtable(
            {
                "single_col": [1.0, 2.0, 3.0],
                "group_col_a": [4.0, 5.0, 6.0],
                "group_col_b": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] single, float[N] group) => (float[N] output) {
                output = Concat <axis: int = 1> (single, group)
            }
        """)

        variables = GraphVariables(
            ibis.memtable({"single": [1.0], "group": [1.0]}), model
        )
        variables["single"] = table["single_col"]
        variables["group"] = ValueVariablesGroup(
            {
                "col_a": table["group_col_a"],
                "col_b": table["group_col_b"],
            }
        )

        translator = ConcatTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 3
        assert "single" in result
        assert "group.col_a" in result
        assert "group.col_b" in result

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["single"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["group.col_a"])) == [4.0, 5.0, 6.0]
        assert list(backend.execute(result["group.col_b"])) == [7.0, 8.0, 9.0]

    def test_concat_unsupported_axis(self):
        """Test error for unsupported axis."""
        table = ibis.memtable(
            {"feature1": [1.0, 2.0, 3.0], "feature2": [4.0, 5.0, 6.0]}
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] feature1, float[N] feature2) => (float[N] output) {
                output = Concat <axis: int = 0> (feature1, feature2)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ConcatTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError, match="only supports concatenating over columns"
        ):
            translator.process()


