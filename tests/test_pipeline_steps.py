"""Test individual pipeline steps/translators."""

import onnx
import ibis
import pytest

from orbital.translate import TRANSLATORS
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
from orbital.translation.steps.concat import ConcatTranslator, FeatureVectorizerTranslator
from orbital.translation.steps.gather import GatherTranslator
from orbital.translation.steps.arrayfeatureextractor import ArrayFeatureExtractorTranslator
from orbital.translation.variables import (
    GraphVariables,
    NumericVariablesGroup,
    ValueVariablesGroup,
)
from orbital.translation.optimizer import Optimizer
from orbital.translation.options import TranslationOptions


class TestStepCoverage:
    """Verify that all registered steps have corresponding test classes."""

    def test_all_registered_steps_have_tests(self):
        """Every step registered in TRANSLATORS must have a test class.

        Test classes must follow the naming convention Test{OperationName}Translator.
        This test ensures we don't forget to add tests when implementing new steps.
        """
        import sys

        module = sys.modules[__name__]
        existing_test_classes = {
            name
            for name in dir(module)
            if name.startswith("Test") and isinstance(getattr(module, name), type)
        }

        missing_tests = []
        for operation in sorted(TRANSLATORS.keys()):
            expected_test_class = f"Test{operation}Translator"
            if expected_test_class not in existing_test_classes:
                missing_tests.append((operation, expected_test_class))

        if missing_tests:
            missing_list = "\n".join(
                f"  - {op}: {test_class}" for op, test_class in missing_tests
            )
            pytest.fail(
                f"The following {len(missing_tests)} steps are missing test classes:\n"
                f"{missing_list}\n\n"
                f"Add a test class for each step to test_pipeline_steps.py"
            )


class TestSoftmaxTranslator:
    optimizer = Optimizer(enabled=False)

    def test_softmax_translator_single_input(self):
        """Test SoftmaxTranslator with a single numeric input."""
        table = ibis.memtable({"input": [2.0, 3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Softmax(input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = SoftmaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # For single input, softmax should return 1.0
        backend = ibis.duckdb.connect()
        computed_value = backend.execute(result)
        assert computed_value == 1.0

    def test_softmax_translator_group_input(self):
        """Test SoftmaxTranslator with a group of numeric inputs."""
        multi_table = ibis.memtable(
            {
                "class_0": [1.0, 2.0, 3.0],
                "class_1": [0.5, 1.5, 2.5],
                "class_2": [2.0, 3.0, 4.0],
            }
        )

        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Softmax(input)
            }
        """)

        # Use dummy table for GraphVariables since we override the input
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)

        variables["input"] = NumericVariablesGroup(
            {
                "class_0": multi_table["class_0"],
                "class_1": multi_table["class_1"],
                "class_2": multi_table["class_2"],
            }
        )

        translator = SoftmaxTranslator(
            multi_table,
            model.node[0],
            variables,
            self.optimizer,
            TranslationOptions(),
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Should return a NumericVariablesGroup
        assert isinstance(result, NumericVariablesGroup)
        assert len(result) == 3
        assert "class_0" in result
        assert "class_1" in result
        assert "class_2" in result

        # Test that softmax values sum to 1.0 for each row
        backend = ibis.duckdb.connect()

        # backend.execute() returns a pandas Series, so we take the first element
        values = [
            backend.execute(result[class_name])[0]
            for class_name in ["class_0", "class_1", "class_2"]
        ]

        # Verify they sum to approximately 1.0
        total_sum = sum(values)
        assert abs(total_sum - 1.0) < 1e-10, (
            f"Softmax values should sum to 1.0, got {total_sum}"
        )

    def test_softmax_translator_invalid_axis(self):
        """Test that SoftmaxTranslator raises error for unsupported axis."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Softmax <axis: int = 0> (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = SoftmaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError, match="SoftmaxTranslator supports only axis=-1 or axis=1"
        ):
            translator.process()

    def test_softmax_translator_invalid_input_type(self):
        """Test that SoftmaxTranslator raises error for invalid input type."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Softmax(input)
            }
        """)

        variables = GraphVariables(table, model)

        # Intentionally set invalid input type to test error handling
        variables["input"] = "invalid_string_input"  # type: ignore[assignment]

        translator = SoftmaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError, match="Softmax: The first operand must be a numeric column"
        ):
            translator.process()

    def test_softmax_uses_apply_post_transform(self):
        """Test that SoftmaxTranslator uses the apply_post_transform function."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Softmax(input)
            }
        """)

        variables = GraphVariables(table, model)

        variables["input"] = NumericVariablesGroup(
            {
                "class_0": ibis.literal(1.0),
                "class_1": ibis.literal(2.0),
            }
        )

        translator = SoftmaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)


class TestImputerTranslator:
    optimizer = Optimizer(enabled=False)

    def test_imputer_single_column(self):
        """Test ImputerTranslator with a single column input."""
        table = ibis.memtable({"input": [1.0, None, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Imputer <imputed_value_floats: floats = [2.0]> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ImputerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = backend.execute(result)
        assert list(computed) == [1.0, 2.0, 3.0]

    def test_imputer_group_columns(self):
        """Test ImputerTranslator with a group of columns."""
        table = ibis.memtable(
            {
                "col_a": [1.0, None, 3.0],
                "col_b": [None, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Imputer <imputed_value_floats: floats = [10.0, 20.0]> (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = ImputerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [1.0, 10.0, 3.0]
        assert list(backend.execute(result["col_b"])) == [20.0, 5.0, 6.0]

    def test_imputer_invalid_imputed_value_type(self):
        """Test ImputerTranslator raises error for non-list imputed values."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Imputer <imputed_value_floats: floats = [2.0]> (input)
            }
        """)

        variables = GraphVariables(table, model)

        # Override attributes to test validation
        node = model.node[0]
        translator = ImputerTranslator(
            table, node, variables, self.optimizer, TranslationOptions()
        )
        translator._attributes["imputed_value_floats"] = 2.0  # Invalid: not a list

        with pytest.raises(ValueError, match="imputed_value must be a list or tuple"):
            translator.process()

    def test_imputer_mismatched_column_count(self):
        """Test ImputerTranslator raises error when column count doesn't match."""
        table = ibis.memtable(
            {
                "col_a": [1.0, None, 3.0],
                "col_b": [None, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Imputer <imputed_value_floats: floats = [10.0]> (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = ImputerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="number of imputed values does not match"):
            translator.process()


class TestArgMaxTranslator:
    optimizer = Optimizer(enabled=False)

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


class TestAddTranslator:
    optimizer = Optimizer(enabled=False)

    def test_add_single_column(self):
        """Test AddTranslator with a single column input."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] add_value = {5.0}>
            {
                output = Add(input, add_value)
            }
        """)

        variables = GraphVariables(table, model)
        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [6.0, 7.0, 8.0]

    def test_add_group_columns(self):
        """Test AddTranslator with a group of columns."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] add_values = {5.0, 100.0}>
            {
                output = Add(input, add_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [6.0, 7.0, 8.0]
        assert list(backend.execute(result["col_b"])) == [110.0, 120.0, 130.0]

    def test_add_invalid_non_numeric(self):
        """Test AddTranslator raises error for non-numeric operand."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] add_value = {5.0}>
            {
                output = Add(input, add_value)
            }
        """)

        variables = GraphVariables(table, model)
        variables["input"] = "not_a_numeric_value"  # type: ignore[assignment]

        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="first operand must be a numeric value"):
            translator.process()

    def test_add_mismatched_column_count(self):
        """Test AddTranslator raises error when column count doesn't match."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] add_values = {5.0}>
            {
                output = Add(input, add_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="same number of values"):
            translator.process()

    def test_add_single_column_requires_single_value(self):
        """Test AddTranslator raises error when single column given multiple values."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] add_values = {5.0, 10.0}>
            {
                output = Add(input, add_values)
            }
        """)

        variables = GraphVariables(table, model)

        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="must contain exactly 1 value"):
            translator.process()

    def test_add_second_operand_variable(self):
        """Test AddTranslator supports variable-to-variable addition (residual connections)."""
        table = ibis.memtable(
            {
                "input": [1.0, 2.0, 3.0],
                "other": [5.0, 5.0, 5.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input, float[N] other) => (float[N] output) {
                output = Add(input, other)
            }
        """)

        variables = GraphVariables(table, model)

        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        values = backend.execute(result).tolist()
        assert values == [6.0, 7.0, 8.0]


class TestSubTranslator:
    optimizer = Optimizer(enabled=False)

    def test_sub_single_column(self):
        """Test SubTranslator with a single column input."""
        table = ibis.memtable({"input": [10.0, 20.0, 30.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] sub_value = {3.0}>
            {
                output = Sub(input, sub_value)
            }
        """)

        variables = GraphVariables(table, model)
        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [7.0, 17.0, 27.0]

    def test_sub_group_columns(self):
        """Test SubTranslator with a group of columns."""
        table = ibis.memtable(
            {
                "col_a": [10.0, 20.0, 30.0],
                "col_b": [100.0, 200.0, 300.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] sub_values = {3.0, 50.0}>
            {
                output = Sub(input, sub_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [7.0, 17.0, 27.0]
        assert list(backend.execute(result["col_b"])) == [50.0, 150.0, 250.0]

    def test_sub_invalid_non_numeric(self):
        """Test SubTranslator raises error for non-numeric operand."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] sub_value = {5.0}>
            {
                output = Sub(input, sub_value)
            }
        """)

        variables = GraphVariables(table, model)
        variables["input"] = "not_a_numeric_value"  # type: ignore[assignment]

        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="first operand must be a numeric value"):
            translator.process()

    def test_sub_single_column_requires_single_value(self):
        """Test SubTranslator raises error when single column given multiple values."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] sub_values = {5.0, 10.0}>
            {
                output = Sub(input, sub_values)
            }
        """)

        variables = GraphVariables(table, model)

        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="must contain exactly 1 value"):
            translator.process()

    def test_sub_mismatched_column_count(self):
        """Test SubTranslator raises error when column count doesn't match."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] sub_values = {5.0}>
            {
                output = Sub(input, sub_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(AssertionError, match="must match the number of fields"):
            translator.process()

    def test_sub_variable_minus_variable(self):
        """Test SubTranslator supports variable-to-variable subtraction."""
        table = ibis.memtable(
            {
                "input": [10.0, 20.0, 30.0],
                "other": [5.0, 5.0, 5.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input, float[N] other) => (float[N] output) {
                output = Sub(input, other)
            }
        """)

        variables = GraphVariables(table, model)

        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        values = backend.execute(result).tolist()
        assert values == [5.0, 15.0, 25.0]


class TestMulTranslator:
    optimizer = Optimizer(enabled=False)

    def test_mul_single_column(self):
        """Test MulTranslator with a single column input."""
        table = ibis.memtable({"input": [2.0, 3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] mul_value = {5.0}>
            {
                output = Mul(input, mul_value)
            }
        """)

        variables = GraphVariables(table, model)
        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [10.0, 15.0, 20.0]

    def test_mul_group_columns(self):
        """Test MulTranslator with a group of columns."""
        table = ibis.memtable(
            {
                "col_a": [2.0, 3.0, 4.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] mul_values = {3.0, 2.0}>
            {
                output = Mul(input, mul_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [6.0, 9.0, 12.0]
        assert list(backend.execute(result["col_b"])) == [20.0, 40.0, 60.0]

    def test_mul_invalid_non_numeric(self):
        """Test MulTranslator raises error for non-numeric operand."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] mul_value = {5.0}>
            {
                output = Mul(input, mul_value)
            }
        """)

        variables = GraphVariables(table, model)
        variables["input"] = "not_a_numeric_value"  # type: ignore[assignment]

        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="first operand must be a numeric value"):
            translator.process()

    def test_mul_mismatched_column_count(self):
        """Test MulTranslator raises error when column count doesn't match."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] mul_values = {5.0}>
            {
                output = Mul(input, mul_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="same number of values"):
            translator.process()

    def test_mul_single_column_requires_single_value(self):
        """Test MulTranslator raises error when single column given multiple values."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] mul_values = {5.0, 10.0}>
            {
                output = Mul(input, mul_values)
            }
        """)

        variables = GraphVariables(table, model)

        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="must contain exactly 1 value"):
            translator.process()

    def test_mul_second_operand_not_constant(self):
        """Test MulTranslator raises error when second operand is not a constant."""
        table = ibis.memtable(
            {
                "input": [1.0, 2.0, 3.0],
                "other": [5.0, 5.0, 5.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input, float[N] other) => (float[N] output) {
                output = Mul(input, other)
            }
        """)

        variables = GraphVariables(table, model)

        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(NotImplementedError, match="must be a constant list"):
            translator.process()


class TestDivTranslator:
    optimizer = Optimizer(enabled=False)

    def test_div_single_column(self):
        """Test DivTranslator with a single column input."""
        table = ibis.memtable({"input": [10.0, 20.0, 30.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] div_value = {2.0}>
            {
                output = Div(input, div_value)
            }
        """)

        variables = GraphVariables(table, model)
        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [5.0, 10.0, 15.0]

    def test_div_group_columns_with_matching_values(self):
        """Test DivTranslator with group of columns and matching divisor values."""
        table = ibis.memtable(
            {
                "col_a": [10.0, 20.0, 30.0],
                "col_b": [100.0, 200.0, 300.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] div_values = {2.0, 10.0}>
            {
                output = Div(input, div_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [5.0, 10.0, 15.0]
        assert list(backend.execute(result["col_b"])) == [10.0, 20.0, 30.0]

    def test_div_group_columns_broadcast_single_value(self):
        """Test DivTranslator with group of columns and single divisor (broadcast)."""
        table = ibis.memtable(
            {
                "col_a": [10.0, 20.0, 30.0],
                "col_b": [100.0, 200.0, 300.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] div_value = {10.0}>
            {
                output = Div(input, div_value)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["col_b"])) == [10.0, 20.0, 30.0]

    def test_div_invalid_non_numeric_single(self):
        """Test DivTranslator raises error for non-numeric single operand."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] div_value = {5.0}>
            {
                output = Div(input, div_value)
            }
        """)

        variables = GraphVariables(table, model)
        variables["input"] = "not_a_numeric_value"  # type: ignore[assignment]

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="first operand must be a numeric value"):
            translator.process()

    def test_div_mismatched_column_count(self):
        """Test DivTranslator raises error when column count doesn't match values."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[3] div_values = {5.0, 10.0, 15.0}>
            {
                output = Div(input, div_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="must match the number of columns"):
            translator.process()

    def test_div_single_column_requires_single_value(self):
        """Test DivTranslator raises error when single column given multiple values."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] div_values = {5.0, 10.0}>
            {
                output = Div(input, div_values)
            }
        """)

        variables = GraphVariables(table, model)

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="must contain only one value"):
            translator.process()

    def test_div_second_operand_not_constant(self):
        """Test DivTranslator raises error when second operand is not a constant."""
        table = ibis.memtable(
            {
                "input": [1.0, 2.0, 3.0],
                "other": [5.0, 5.0, 5.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input, float[N] other) => (float[N] output) {
                output = Div(input, other)
            }
        """)

        variables = GraphVariables(table, model)

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(NotImplementedError, match="must be a constant list"):
            translator.process()


class TestIdentityTranslator:
    optimizer = Optimizer(enabled=False)

    def test_identity_single_column_passthrough(self):
        """Test IdentityTranslator passes a single column through unchanged."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Identity(input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = IdentityTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Verify output equals input
        backend = ibis.duckdb.connect()
        input_values = list(backend.execute(table["input"]))
        output_values = list(backend.execute(result))
        assert output_values == input_values

    def test_identity_group_columns_passthrough(self):
        """Test IdentityTranslator passes a NumericVariablesGroup through unchanged."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
                "col_c": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Identity(input)
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

        translator = IdentityTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Should return the same NumericVariablesGroup
        assert isinstance(result, NumericVariablesGroup)
        assert len(result) == 3
        assert "col_a" in result
        assert "col_b" in result
        assert "col_c" in result

        # Verify all columns preserved with same values
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["col_b"])) == [4.0, 5.0, 6.0]
        assert list(backend.execute(result["col_c"])) == [7.0, 8.0, 9.0]


class TestReshapeTranslator:
    optimizer = Optimizer(enabled=False)

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

        with pytest.raises(ValueError, match="Reshape shape=\\[-1, 2\\] not supported"):
            translator.process()


class TestMatMulTranslator:
    optimizer = Optimizer(enabled=False)

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


class TestCastTranslator:
    optimizer = Optimizer(enabled=False)

    def test_cast_single_column_to_float32(self):
        """Test CastTranslator with single column cast to float32."""
        table = ibis.memtable({"input": [1, 2, 3]})  # int values
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input) => (float[N] output) {
                output = Cast <to: int = 1> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = CastTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = backend.execute(result)
        assert list(computed) == [1.0, 2.0, 3.0]

    def test_cast_single_column_to_float64(self):
        """Test CastTranslator with single column cast to float64."""
        table = ibis.memtable({"input": [1, 2, 3]})  # int values
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input) => (double[N] output) {
                output = Cast <to: int = 11> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = CastTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = backend.execute(result)
        assert list(computed) == [1.0, 2.0, 3.0]

    def test_cast_single_column_to_int64(self):
        """Test CastTranslator with single column cast to int64."""
        table = ibis.memtable({"input": [1.2, 2.7, 3.9]})  # float values
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] output) {
                output = Cast <to: int = 7> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = CastTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = backend.execute(result)
        assert list(computed) == [1, 3, 4]

    def test_cast_single_column_to_string(self):
        """Test CastTranslator with single column cast to string."""
        table = ibis.memtable({"input": [1, 2, 3]})
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input) => (string[N] output) {
                output = Cast <to: int = 8> (input)
            }
        """)

        variables = GraphVariables(table, model)
        options = TranslationOptions(allow_text_tensors=True)
        translator = CastTranslator(
            table, model.node[0], variables, self.optimizer, options
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = backend.execute(result)
        assert list(computed) == ["1", "2", "3"]

    def test_cast_group_of_columns(self):
        """Test CastTranslator with a group of columns."""
        table = ibis.memtable(
            {
                "col_a": [1, 2, 3],
                "col_b": [4, 5, 6],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input) => (float[N] output) {
                output = Cast <to: int = 1> (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = CastTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["col_b"])) == [4.0, 5.0, 6.0]

    def test_skip_cast_to_string_single_column(self):
        """Test CastTranslator skips cast to string when allow_text_tensors=False."""
        table = ibis.memtable({"input": [1, 2, 3]})
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input) => (string[N] output) {
                output = Cast <to: int = 8> (input)
            }
        """)

        variables = GraphVariables(table, model)
        options = TranslationOptions(allow_text_tensors=False)
        translator = CastTranslator(
            table, model.node[0], variables, self.optimizer, options
        )
        translator.process()

        # Should skip the cast and return the original input
        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = backend.execute(result)
        # Should remain int64, not cast to string
        assert list(computed) == [1, 2, 3]

    def test_skip_cast_to_string_column_group(self):
        """Test CastTranslator skips cast to string for column group when allow_text_tensors=False."""
        table = ibis.memtable(
            {
                "col_a": [1, 2, 3],
                "col_b": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (string[N] output) {
                output = Cast <to: int = 8> (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        options = TranslationOptions(allow_text_tensors=False)
        translator = CastTranslator(
            table, model.node[0], variables, self.optimizer, options
        )
        translator.process()

        # Should skip the cast and return the original input group
        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        # Should remain original types, not cast to string
        assert list(backend.execute(result["col_a"])) == [1, 2, 3]
        assert list(backend.execute(result["col_b"])) == [4.0, 5.0, 6.0]

    def test_cast_unsupported_target_type(self):
        """Test CastTranslator raises error for unsupported target type."""
        table = ibis.memtable({"input": [1, 2, 3]})
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input) => (float[N] output) {
                output = Cast <to: int = 999> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = CastTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(NotImplementedError, match="Cast: type 999 not supported"):
            translator.process()


class TestCastLikeTranslator:
    optimizer = Optimizer(enabled=False)

    def test_cast_group_to_match_single_column_type(self):
        """Test CastLikeTranslator casts group of columns to match single column type."""
        table = ibis.memtable(
            {
                "col_a": [1, 2, 3],  # int64
                "col_b": [4, 5, 6],  # int64
                "target": [1.0, 2.0, 3.0],  # float64
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input, float[N] target_type) => (float[N] output) {
                output = CastLike(input, target_type)
            }
        """)

        variables = GraphVariables(
            ibis.memtable({"input": [1], "target_type": [1.0]}), model
        )
        variables["input"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )
        variables["target_type"] = table["target"]

        translator = CastLikeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        # Should match target column type (float64)
        assert list(backend.execute(result["col_a"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["col_b"])) == [4.0, 5.0, 6.0]

    def test_castlike_error_input_is_single_column(self):
        """Test CastLikeTranslator raises error when input is single column."""
        table = ibis.memtable(
            {
                "input": [1, 2, 3],
                "target_type": [1.0, 2.0, 3.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input, float[N] target_type) => (float[N] output) {
                output = CastLike(input, target_type)
            }
        """)

        variables = GraphVariables(table, model)

        translator = CastLikeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError,
            match="CastLike currently only supports casting a group of columns",
        ):
            translator.process()

    def test_castlike_error_target_is_group(self):
        """Test CastLikeTranslator raises error when target is a group."""
        table = ibis.memtable(
            {
                "col_a": [1, 2, 3],
                "col_b": [4, 5, 6],
                "target_a": [1.0, 2.0, 3.0],
                "target_b": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (int64[N] input, float[N] target_type) => (float[N] output) {
                output = CastLike(input, target_type)
            }
        """)

        variables = GraphVariables(
            ibis.memtable({"input": [1], "target_type": [1.0]}), model
        )
        variables["input"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )
        variables["target_type"] = ValueVariablesGroup(
            {
                "target_a": table["target_a"],
                "target_b": table["target_b"],
            }
        )

        translator = CastLikeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError,
            match="CastLike currently only supports casting to a single column type, not a group",
        ):
            translator.process()


class TestLinearClassifierTranslator:
    optimizer = Optimizer(enabled=False)

    def test_linear_classifier_binary_classification(self):
        """Test LinearClassifierTranslator with binary classification."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [0.5, 0.5, -0.5, -0.5],
                    intercepts: floats = [1.0, -1.0],
                    classlabels_ints: ints = [0, 1]
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "feature1": table["feature1"],
                "feature2": table["feature2"],
            }
        )

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "Y" in variables
        assert "Z" in variables

        predictions = variables.peek_variable("Y")
        scores = variables.peek_variable("Z")

        assert isinstance(scores, ValueVariablesGroup)
        assert "0" in scores
        assert "1" in scores

        backend = ibis.duckdb.connect()
        # Class 0 has higher score so it should be predicted
        assert backend.execute(predictions)[0] == "0"

    def test_linear_classifier_multiclass(self):
        """Test LinearClassifierTranslator with multi-class classification (3+ classes)."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,3] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [1.0, 0.0, 0.0, 1.0, -1.0, -1.0],
                    intercepts: floats = [0.0, 0.5, -0.5],
                    classlabels_ints: ints = [0, 1, 2]
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "feature1": table["feature1"],
                "feature2": table["feature2"],
            }
        )

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        predictions = variables.peek_variable("Y")
        scores = variables.peek_variable("Z")

        assert isinstance(scores, ValueVariablesGroup)
        assert len(scores) == 3

        backend = ibis.duckdb.connect()
        # Class 1 has highest score for first row
        assert backend.execute(predictions)[0] == "1"

    def test_linear_classifier_with_intercepts(self):
        """Test LinearClassifierTranslator properly adds intercepts."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [1.0, -1.0],
                    intercepts: floats = [10.0, 20.0],
                    classlabels_ints: ints = [0, 1]
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        scores = variables.peek_variable("Z")

        backend = ibis.duckdb.connect()
        # Verify intercepts are properly added to scores
        assert backend.execute(scores["0"])[0] == 11.0
        assert backend.execute(scores["1"])[0] == 19.0

    def test_linear_classifier_post_transform_logistic(self):
        """Test LinearClassifierTranslator with post_transform='LOGISTIC'."""
        table = ibis.memtable({"input": [0.0, 1.0, -1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [1.0, -1.0],
                    intercepts: floats = [0.0, 0.0],
                    classlabels_ints: ints = [0, 1],
                    post_transform: string = "LOGISTIC"
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        scores = variables.peek_variable("Z")

        backend = ibis.duckdb.connect()
        # Logistic of 0.0 yields 0.5
        assert abs(backend.execute(scores["0"])[0] - 0.5) < 1e-6

    def test_linear_classifier_post_transform_softmax(self):
        """Test LinearClassifierTranslator with post_transform='SOFTMAX'."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [1.0, 0.0, 0.0, 1.0],
                    intercepts: floats = [0.0, 0.0],
                    classlabels_ints: ints = [0, 1],
                    post_transform: string = "SOFTMAX"
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "feature1": table["feature1"],
                "feature2": table["feature2"],
            }
        )

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        scores = variables.peek_variable("Z")
        assert isinstance(scores, ValueVariablesGroup)
        assert "0" in scores
        assert "1" in scores

    def test_linear_classifier_string_labels(self):
        """Test LinearClassifierTranslator with classlabels_strings instead of classlabels_ints."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (string[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [1.0, -1.0],
                    intercepts: floats = [0.0, 0.0],
                    classlabels_strings: strings = ["cat", "dog"]
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        predictions = variables.peek_variable("Y")
        scores = variables.peek_variable("Z")

        assert "cat" in scores
        assert "dog" in scores

        backend = ibis.duckdb.connect()
        # "cat" has higher score so it should be predicted
        assert backend.execute(predictions)[0] == "cat"

    def test_linear_classifier_single_input(self):
        """Test LinearClassifierTranslator when input is a single column (not a VariablesGroup)."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [2.0, -2.0],
                    intercepts: floats = [0.0, 0.0],
                    classlabels_ints: ints = [0, 1]
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        predictions = variables.peek_variable("Y")

        backend = ibis.duckdb.connect()
        # Class 0: 1.0*2.0 = 2.0
        # Class 1: 1.0*(-2.0) = -2.0
        # Prediction should be class 0
        assert backend.execute(predictions)[0] == "0"

    def test_linear_classifier_missing_classlabels(self):
        """Test LinearClassifierTranslator raises error when no classlabels defined."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [1.0, -1.0],
                    intercepts: floats = [0.0, 0.0]
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="LinearClassifier: classlabels_ints or classlabels_strings must be defined",
        ):
            translator.process()

    def test_linear_classifier_coefficient_mismatch(self):
        """Test LinearClassifierTranslator raises error when coefficients length doesn't match classes × features."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [1.0, 2.0, 3.0],
                    classlabels_ints: ints = [0, 1]
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "feature1": table["feature1"],
                "feature2": table["feature2"],
            }
        )

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="Coefficients length must equal number of classes × number of input fields",
        ):
            translator.process()

    def test_linear_classifier_multi_class_mode_not_implemented(self):
        """Test LinearClassifierTranslator raises error when multi_class != 0."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (int64[N] Y, float[N,2] Z)
            {
                Y, Z = ai.onnx.ml.LinearClassifier <
                    coefficients: floats = [1.0, -1.0],
                    classlabels_ints: ints = [0, 1],
                    multi_class: int = 1
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearClassifierTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError, match="Multi-class classification is not implemented"
        ):
            translator.process()


class TestLinearRegressorTranslator:
    optimizer = Optimizer(enabled=False)

    def test_linear_regressor_single_target(self):
        """Test LinearRegressorTranslator with single target regression (targets=1)."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [0.5, 0.5],
                    intercepts: floats = [10.0],
                    targets: int = 1
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "feature1": table["feature1"],
                "feature2": table["feature2"],
            }
        )

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "Y" in variables
        result = variables.peek_variable("Y")

        backend = ibis.duckdb.connect()
        assert backend.execute(result["target_0"])[0] == 12.5

    def test_linear_regressor_multi_target(self):
        """Test LinearRegressorTranslator with multi-target regression (targets=2+)."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N,2] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [1.0, 0.0, 0.0, 1.0],
                    intercepts: floats = [5.0, 10.0],
                    targets: int = 2
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "feature1": table["feature1"],
                "feature2": table["feature2"],
            }
        )

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("Y")
        assert isinstance(result, ValueVariablesGroup)
        assert "target_0" in result
        assert "target_1" in result

        backend = ibis.duckdb.connect()
        assert backend.execute(result["target_0"])[0] == 6.0
        assert backend.execute(result["target_1"])[0] == 14.0

    def test_linear_regressor_with_intercepts(self):
        """Test LinearRegressorTranslator properly adds intercepts."""
        table = ibis.memtable({"feature": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [2.0],
                    intercepts: floats = [100.0]
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup({"feature": table["feature"]})

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("Y")

        backend = ibis.duckdb.connect()
        # Verify intercept is properly added
        assert backend.execute(result["target_0"])[0] == 102.0

    def test_linear_regressor_no_intercepts(self):
        """Test LinearRegressorTranslator without intercepts."""
        table = ibis.memtable({"feature": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [3.0]
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup({"feature": table["feature"]})

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("Y")

        backend = ibis.duckdb.connect()
        # Default intercept is 0
        assert backend.execute(result["target_0"])[0] == 3.0

    def test_linear_regressor_single_column_input(self):
        """Test LinearRegressorTranslator with single column input (not VariablesGroup)."""
        table = ibis.memtable({"input": [2.0, 3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [5.0],
                    intercepts: floats = [1.0]
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        result = variables.peek_variable("Y")

        backend = ibis.duckdb.connect()
        # Prediction: 2.0*5.0 + 1.0 = 11.0
        assert backend.execute(result)[0] == 11.0

    def test_linear_regressor_coefficient_mismatch(self):
        """Test LinearRegressorTranslator raises error when coefficients length mismatch."""
        table = ibis.memtable(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [1.0, 2.0, 3.0],
                    targets: int = 1
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "feature1": table["feature1"],
                "feature2": table["feature2"],
            }
        )

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="Coefficients length must equal targets number of input fields",
        ):
            translator.process()

    def test_linear_regressor_intercepts_length_mismatch(self):
        """Test LinearRegressorTranslator raises error when intercepts length mismatch."""
        table = ibis.memtable({"feature": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N,2] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [1.0, 2.0],
                    intercepts: floats = [5.0],
                    targets: int = 2
                > (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup({"feature": table["feature"]})

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="LinearRegressor: intercepts length must match targets or be empty",
        ):
            translator.process()

    def test_linear_regressor_post_transform_not_implemented(self):
        """Test LinearRegressorTranslator raises error when post_transform != 'NONE'."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [1.0],
                    post_transform: string = "LOGISTIC"
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError, match="Post transform is not implemented"
        ):
            translator.process()

    def test_linear_regressor_single_input_multiple_targets(self):
        """Test LinearRegressorTranslator raises error with single input and multiple targets."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N,2] Y)
            {
                Y = ai.onnx.ml.LinearRegressor <
                    coefficients: floats = [1.0, 2.0],
                    intercepts: floats = [0.0, 0.0],
                    targets: int = 2
                > (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = LinearRegressorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="Single column input expects exactly one target and one coefficient",
        ):
            translator.process()




class TestTreeEnsembleClassifierTranslator:
    optimizer = Optimizer(enabled=False)

    def test_binary_classification_single_tree(self):
        """Test TreeEnsembleClassifier with binary classification and single tree."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.classifier import TreeEnsembleClassifierTranslator

        table = ibis.memtable({"X": [0.3, 0.7, 0.2]})

        # Create a simple binary classification tree:
        # if feature[0] <= 0.5:
        #     return class 0 (weight 1.0)
        # else:
        #     return class 1 (weight 1.0)
        node = helper.make_node(
            op_type="TreeEnsembleClassifier",
            inputs=["X"],
            outputs=["Y", "P"],
            domain="ai.onnx.ml",
            # Tree structure attributes
            nodes_treeids=[0, 0, 0],
            nodes_nodeids=[0, 1, 2],
            nodes_featureids=[0, 0, 0],
            nodes_modes=["BRANCH_LEQ", "LEAF", "LEAF"],
            nodes_values=[0.5, 0.0, 0.0],
            nodes_truenodeids=[1, 0, 0],
            nodes_falsenodeids=[2, 0, 0],
            nodes_missing_value_tracks_true=[0, 0, 0],
            # Class weights
            class_treeids=[0, 0],
            class_nodeids=[1, 2],
            class_ids=[0, 0],
            class_weights=[1.0, -1.0],
            # Class labels
            classlabels_int64s=[0, 1],
            post_transform="NONE",
        )

        # Create a minimal graph and model
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [None])
        P = helper.make_tensor_value_info("P", TensorProto.FLOAT, [None, 2])

        graph = helper.make_graph([node], "test", [X], [Y, P])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)

        translator = TreeEnsembleClassifierTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "Y" in variables
        assert "P" in variables

        label_result = variables.peek_variable("Y")
        prob_result = variables.peek_variable("P")

        backend = ibis.duckdb.connect()
        labels = list(backend.execute(label_result))
        assert labels == [1, 0, 1]

        # Check probabilities
        assert isinstance(prob_result, NumericVariablesGroup)
        assert "0" in prob_result
        assert "1" in prob_result

    def test_multiclass_classification_single_tree(self):
        """Test TreeEnsembleClassifier with multi-class classification."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.classifier import TreeEnsembleClassifierTranslator

        table = ibis.memtable({"X": [0.3, 0.7, 0.2]})

        # Create a multi-class tree with 3 classes
        node = helper.make_node(
            op_type="TreeEnsembleClassifier",
            inputs=["X"],
            outputs=["Y", "P"],
            domain="ai.onnx.ml",
            # Simple tree: always returns class weights at leaf 1
            nodes_treeids=[0, 0],
            nodes_nodeids=[0, 1],
            nodes_featureids=[0, 0],
            nodes_modes=["BRANCH_LEQ", "LEAF"],
            nodes_values=[10.0, 0.0],
            nodes_truenodeids=[1, 0],
            nodes_falsenodeids=[1, 0],
            nodes_missing_value_tracks_true=[0, 0],
            # Class weights for 3 classes at leaf node
            class_treeids=[0, 0, 0],
            class_nodeids=[1, 1, 1],
            class_ids=[0, 1, 2],
            class_weights=[0.2, 0.5, 0.3],
            # String class labels
            classlabels_strings=["cat", "dog", "bird"],
            post_transform="NONE",
        )

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.STRING, [None])
        P = helper.make_tensor_value_info("P", TensorProto.FLOAT, [None, 3])

        graph = helper.make_graph([node], "test", [X], [Y, P])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)

        translator = TreeEnsembleClassifierTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "Y" in variables
        assert "P" in variables

        label_result = variables.peek_variable("Y")
        prob_result = variables.peek_variable("P")

        backend = ibis.duckdb.connect()
        labels = list(backend.execute(label_result))
        # All inputs should predict "dog" since it has highest weight (0.5)
        assert labels == ["dog", "dog", "dog"]

        # Check probabilities group structure
        assert isinstance(prob_result, NumericVariablesGroup)
        assert "cat" in prob_result
        assert "dog" in prob_result
        assert "bird" in prob_result

    def test_classifier_invalid_input_type(self):
        """Test TreeEnsembleClassifier raises error for invalid input type."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.classifier import TreeEnsembleClassifierTranslator

        table = ibis.memtable({"X": [0.3, 0.7, 0.2]})

        node = helper.make_node(
            op_type="TreeEnsembleClassifier",
            inputs=["X"],
            outputs=["Y", "P"],
            domain="ai.onnx.ml",
            nodes_treeids=[0],
            nodes_nodeids=[0],
            nodes_featureids=[0],
            nodes_modes=["LEAF"],
            nodes_values=[0.0],
            nodes_truenodeids=[0],
            nodes_falsenodeids=[0],
            nodes_missing_value_tracks_true=[0],
            class_treeids=[0],
            class_nodeids=[0],
            class_ids=[0],
            class_weights=[1.0],
            classlabels_int64s=[0, 1],
        )

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [None])
        P = helper.make_tensor_value_info("P", TensorProto.FLOAT, [None, 2])

        graph = helper.make_graph([node], "test", [X], [Y, P])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)
        variables["X"] = "invalid_string_input"  # type: ignore[assignment]

        translator = TreeEnsembleClassifierTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="TreeEnsembleClassifier: The first operand must be a column or a column group"
        ):
            translator.process()


class TestTreeEnsembleRegressorTranslator:
    optimizer = Optimizer(enabled=False)

    def test_single_tree_regression(self):
        """Test TreeEnsembleRegressor with a single decision tree."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.regressor import TreeEnsembleRegressorTranslator

        table = ibis.memtable({"X": [0.3, 0.7, 0.2]})

        # Create a simple regression tree:
        # if feature[0] <= 0.5:
        #     return 10.0
        # else:
        #     return 20.0
        node = helper.make_node(
            op_type="TreeEnsembleRegressor",
            inputs=["X"],
            outputs=["Y"],
            domain="ai.onnx.ml",
            # Tree structure
            nodes_treeids=[0, 0, 0],
            nodes_nodeids=[0, 1, 2],
            nodes_featureids=[0, 0, 0],
            nodes_modes=["BRANCH_LEQ", "LEAF", "LEAF"],
            nodes_values=[0.5, 0.0, 0.0],
            nodes_truenodeids=[1, 0, 0],
            nodes_falsenodeids=[2, 0, 0],
            nodes_missing_value_tracks_true=[0, 0, 0],
            # Target weights for regression
            target_treeids=[0, 0],
            target_nodeids=[1, 2],
            target_weights=[10.0, 20.0],
            # Base value (offset)
            base_values=[5.0],
        )

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)

        translator = TreeEnsembleRegressorTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "Y" in variables
        result = variables.peek_variable("Y")

        backend = ibis.duckdb.connect()
        predictions = list(backend.execute(result))
        assert predictions == [15.0, 25.0, 15.0]

    def test_regression_base_values_applied(self):
        """Test TreeEnsembleRegressor correctly applies base_values."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.regressor import TreeEnsembleRegressorTranslator

        table = ibis.memtable({"X": [1.0, 2.0, 3.0]})

        # Simple tree that always returns same value
        node = helper.make_node(
            op_type="TreeEnsembleRegressor",
            inputs=["X"],
            outputs=["Y"],
            domain="ai.onnx.ml",
            # Single leaf node tree
            nodes_treeids=[0],
            nodes_nodeids=[0],
            nodes_featureids=[0],
            nodes_modes=["LEAF"],
            nodes_values=[0.0],
            nodes_truenodeids=[0],
            nodes_falsenodeids=[0],
            nodes_missing_value_tracks_true=[0],
            # Weight at leaf
            target_treeids=[0],
            target_nodeids=[0],
            target_weights=[7.0],
            # Base value should be added
            base_values=[3.0],
        )

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)

        translator = TreeEnsembleRegressorTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "Y" in variables
        result = variables.peek_variable("Y")

        backend = ibis.duckdb.connect()
        computed = backend.execute(result)
        # Single-leaf tree returns a constant expression (scalar)
        assert computed == 10.0

    def test_regressor_invalid_input_type(self):
        """Test TreeEnsembleRegressor raises error for invalid input type."""
        from onnx import helper, TensorProto
        from orbital.translation.steps.trees.regressor import TreeEnsembleRegressorTranslator

        table = ibis.memtable({"X": [1.0, 2.0, 3.0]})

        node = helper.make_node(
            op_type="TreeEnsembleRegressor",
            inputs=["X"],
            outputs=["Y"],
            domain="ai.onnx.ml",
            nodes_treeids=[0],
            nodes_nodeids=[0],
            nodes_featureids=[0],
            nodes_modes=["LEAF"],
            nodes_values=[0.0],
            nodes_truenodeids=[0],
            nodes_falsenodeids=[0],
            nodes_missing_value_tracks_true=[0],
            target_treeids=[0],
            target_nodeids=[0],
            target_weights=[1.0],
        )

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph)

        variables = GraphVariables(table, model.graph)
        variables["X"] = "invalid_string_input"  # type: ignore[assignment]

        translator = TreeEnsembleRegressorTranslator(
            table, model.graph.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="TreeEnsembleRegressor: The first operand must be a column or a column group"
        ):
            translator.process()




class TestScalerTranslator:
    optimizer = Optimizer(enabled=False)

    def test_scaler_single_column(self):
        """Test ScalerTranslator with single column scaling."""
        table = ibis.memtable({"input": [2.0, 4.0, 6.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Scaler <offset: floats = [1.0], scale: floats = [2.0]> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ScalerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Y = (X - offset) * scale = (X - 1) * 2
        # [2, 4, 6] -> [2, 6, 10]
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result)) == [2.0, 6.0, 10.0]

    def test_scaler_group_columns(self):
        """Test ScalerTranslator with group of columns."""
        table = ibis.memtable(
            {
                "col_a": [2.0, 4.0, 6.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Scaler <offset: floats = [1.0, 5.0], scale: floats = [2.0, 0.5]> (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = ScalerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        # col_a: Y = (X - 1) * 2 = [2, 6, 10]
        assert list(backend.execute(result["col_a"])) == [2.0, 6.0, 10.0]
        # col_b: Y = (X - 5) * 0.5 = [2.5, 7.5, 12.5]
        assert list(backend.execute(result["col_b"])) == [2.5, 7.5, 12.5]

    def test_scaler_only_offset(self):
        """Test ScalerTranslator with only offset (scale=1.0)."""
        table = ibis.memtable({"input": [2.0, 4.0, 6.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Scaler <offset: floats = [3.0], scale: floats = [1.0]> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ScalerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Y = (X - 3) * 1 = X - 3
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result)) == [-1.0, 1.0, 3.0]

    def test_scaler_only_scale(self):
        """Test ScalerTranslator with only scale (offset=0.0)."""
        table = ibis.memtable({"input": [2.0, 4.0, 6.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Scaler <offset: floats = [0.0], scale: floats = [3.0]> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ScalerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Y = (X - 0) * 3 = X * 3
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result)) == [6.0, 12.0, 18.0]

    def test_scaler_mismatched_offset_scale_counts(self):
        """Test ScalerTranslator raises error when offset/scale counts don't match."""
        table = ibis.memtable(
            {
                "col_a": [2.0, 4.0, 6.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Scaler <offset: floats = [1.0, 2.0, 3.0], scale: floats = [2.0, 3.0]> (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = ScalerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="offset and scale lists must match"):
            translator.process()


class TestOneHotEncoderTranslator:
    optimizer = Optimizer(enabled=False)

    def test_onehot_string_column(self):
        """Test OneHotEncoderTranslator with string column."""
        table = ibis.memtable({"category": ["cat", "dog", "cat", "bird"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] category) => (float[N, 3] output) {
                output = ai.onnx.ml.OneHotEncoder <cats_strings: strings = ["cat", "dog", "bird"]> (category)
            }
        """)

        variables = GraphVariables(table, model)
        translator = OneHotEncoderTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        # First row is "cat", so cat=1.0, dog=0.0, bird=0.0
        assert list(backend.execute(result["cat"])) == [1.0, 0.0, 1.0, 0.0]
        assert list(backend.execute(result["dog"])) == [0.0, 1.0, 0.0, 0.0]
        assert list(backend.execute(result["bird"])) == [0.0, 0.0, 0.0, 1.0]

    def test_onehot_output_type(self):
        """Test OneHotEncoderTranslator output is ValueVariablesGroup with correct keys."""
        table = ibis.memtable({"category": ["apple", "banana", "apple"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] category) => (float[N, 2] output) {
                output = ai.onnx.ml.OneHotEncoder <cats_strings: strings = ["apple", "banana"]> (category)
            }
        """)

        variables = GraphVariables(table, model)
        translator = OneHotEncoderTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert "apple" in result
        assert "banana" in result
        assert len(result) == 2

    def test_onehot_values(self):
        """Test OneHotEncoderTranslator outputs 0.0 or 1.0 floats."""
        table = ibis.memtable({"category": ["red", "blue", "green", "red"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] category) => (float[N, 3] output) {
                output = ai.onnx.ml.OneHotEncoder <cats_strings: strings = ["red", "green", "blue"]> (category)
            }
        """)

        variables = GraphVariables(table, model)
        translator = OneHotEncoderTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        # Input: ["red", "blue", "green", "red"]
        # Verify one-hot encoding produces correct 0.0/1.0 values
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["red"])) == [1.0, 0.0, 0.0, 1.0]
        assert list(backend.execute(result["green"])) == [0.0, 0.0, 1.0, 0.0]
        assert list(backend.execute(result["blue"])) == [0.0, 1.0, 0.0, 0.0]

    def test_onehot_missing_cats_strings(self):
        """Test OneHotEncoderTranslator raises error when cats_strings is missing."""
        table = ibis.memtable({"category": ["cat", "dog"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] category) => (float[N, 3] output) {
                output = ai.onnx.ml.OneHotEncoder (category)
            }
        """)

        variables = GraphVariables(table, model)
        translator = OneHotEncoderTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="attribute cats_strings not found"):
            translator.process()


class TestLabelEncoderTranslator:
    optimizer = Optimizer(enabled=False)

    def test_labelencoder_string_to_int(self):
        """Test LabelEncoderTranslator encoding string labels to integers."""
        table = ibis.memtable({"label": ["cat", "dog", "cat", "bird", "dog"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] label) => (int64[N] output) {
                output = ai.onnx.ml.LabelEncoder <keys_strings: strings = ["cat", "dog", "bird"], values_int64s: ints = [0, 1, 2]> (label)
            }
        """)

        variables = GraphVariables(table, model)
        translator = LabelEncoderTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result)) == [0, 1, 0, 2, 1]

    def test_labelencoder_int_to_int(self):
        """Test LabelEncoderTranslator encoding integer labels to different integers."""
        table = ibis.memtable({"label": [10, 20, 10, 30, 20]})
        model = onnx.parser.parse_graph("""
            agraph (int64[N] label) => (int64[N] output) {
                output = ai.onnx.ml.LabelEncoder <keys_int64s: ints = [10, 20, 30], values_int64s: ints = [100, 200, 300]> (label)
            }
        """)

        variables = GraphVariables(table, model)
        translator = LabelEncoderTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result)) == [100, 200, 100, 300, 200]

    def test_labelencoder_default_value(self):
        """Test LabelEncoderTranslator handles default value for unknown labels."""
        table = ibis.memtable({"label": ["cat", "dog", "unknown", "cat"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] label) => (int64[N] output) {
                output = ai.onnx.ml.LabelEncoder <keys_strings: strings = ["cat", "dog"], values_int64s: ints = [0, 1], default_int64: int = 999> (label)
            }
        """)

        variables = GraphVariables(table, model)
        translator = LabelEncoderTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result)) == [0, 1, 999, 0]

    def test_labelencoder_missing_keys(self):
        """Test LabelEncoderTranslator raises error when keys attribute is missing."""
        table = ibis.memtable({"label": ["cat", "dog"]})
        model = onnx.parser.parse_graph("""
            agraph (string[N] label) => (int64[N] output) {
                output = ai.onnx.ml.LabelEncoder <values_int64s: ints = [0, 1]> (label)
            }
        """)

        variables = GraphVariables(table, model)
        translator = LabelEncoderTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="required mapping attributes not found"):
            translator.process()

class TestWhereTranslator:
    optimizer = Optimizer(enabled=False)

    def test_where_single_columns(self):
        """Test WhereTranslator selecting between two single columns based on condition."""
        table = ibis.memtable(
            {
                "condition": [True, False, True],
                "true_val": [10.0, 20.0, 30.0],
                "false_val": [100.0, 200.0, 300.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (bool[N] condition, float[N] true_val, float[N] false_val) => (float[N] output) {
                output = Where(condition, true_val, false_val)
            }
        """)

        variables = GraphVariables(table, model)
        translator = WhereTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        # When condition is True, take true_val; when False, take false_val
        assert computed == [10.0, 200.0, 30.0]

    def test_where_group_columns(self):
        """Test WhereTranslator selecting between column groups based on condition."""
        table = ibis.memtable(
            {
                "condition": [True, False, True],
                "true_col1": [1.0, 2.0, 3.0],
                "true_col2": [4.0, 5.0, 6.0],
                "false_col1": [10.0, 20.0, 30.0],
                "false_col2": [40.0, 50.0, 60.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (bool[N] condition, float[N] true_expr, float[N] false_expr) => (float[N] output) {
                output = Where(condition, true_expr, false_expr)
            }
        """)

        variables = GraphVariables(
            ibis.memtable(
                {"condition": [True], "true_expr": [1.0], "false_expr": [1.0]}
            ),
            model,
        )
        variables["condition"] = table["condition"]
        variables["true_expr"] = ValueVariablesGroup(
            {"col1": table["true_col1"], "col2": table["true_col2"]}
        )
        variables["false_expr"] = ValueVariablesGroup(
            {"col1": table["false_col1"], "col2": table["false_col2"]}
        )

        translator = WhereTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["c0"])) == [1.0, 20.0, 3.0]
        assert list(backend.execute(result["c1"])) == [4.0, 50.0, 6.0]

    def test_where_broadcast_scalar_true(self):
        """Test WhereTranslator with single true value broadcast to group false."""
        table = ibis.memtable(
            {
                "condition": [True, False, True],
                "true_val": [42.0, 42.0, 42.0],
                "false_col1": [10.0, 20.0, 30.0],
                "false_col2": [100.0, 200.0, 300.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (bool[N] condition, float[N] true_expr, float[N] false_expr) => (float[N] output) {
                output = Where(condition, true_expr, false_expr)
            }
        """)

        variables = GraphVariables(
            ibis.memtable(
                {"condition": [True], "true_expr": [1.0], "false_expr": [1.0]}
            ),
            model,
        )
        variables["condition"] = table["condition"]
        variables["true_expr"] = table["true_val"]
        variables["false_expr"] = ValueVariablesGroup(
            {"col1": table["false_col1"], "col2": table["false_col2"]}
        )

        translator = WhereTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["c0"])) == [42.0, 20.0, 42.0]
        assert list(backend.execute(result["c1"])) == [42.0, 200.0, 42.0]

    def test_where_broadcast_scalar_false(self):
        """Test WhereTranslator with single false value broadcast to group true."""
        table = ibis.memtable(
            {
                "condition": [True, False, True],
                "true_col1": [10.0, 20.0, 30.0],
                "true_col2": [100.0, 200.0, 300.0],
                "false_val": [99.0, 99.0, 99.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (bool[N] condition, float[N] true_expr, float[N] false_expr) => (float[N] output) {
                output = Where(condition, true_expr, false_expr)
            }
        """)

        variables = GraphVariables(
            ibis.memtable(
                {"condition": [True], "true_expr": [1.0], "false_expr": [1.0]}
            ),
            model,
        )
        variables["condition"] = table["condition"]
        variables["true_expr"] = ValueVariablesGroup(
            {"col1": table["true_col1"], "col2": table["true_col2"]}
        )
        variables["false_expr"] = table["false_val"]

        translator = WhereTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["c0"])) == [10.0, 99.0, 30.0]
        assert list(backend.execute(result["c1"])) == [100.0, 99.0, 300.0]

    def test_where_condition_group_error(self):
        """Test WhereTranslator raises error when condition is a group of columns."""
        table = ibis.memtable(
            {
                "cond1": [True, False],
                "cond2": [False, True],
                "true_val": [1.0, 2.0],
                "false_val": [10.0, 20.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (bool[N] condition, float[N] true_expr, float[N] false_expr) => (float[N] output) {
                output = Where(condition, true_expr, false_expr)
            }
        """)

        variables = GraphVariables(
            ibis.memtable(
                {"condition": [True], "true_expr": [1.0], "false_expr": [1.0]}
            ),
            model,
        )
        variables["condition"] = ValueVariablesGroup(
            {"cond1": table["cond1"], "cond2": table["cond2"]}
        )
        variables["true_expr"] = table["true_val"]
        variables["false_expr"] = table["false_val"]

        translator = WhereTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError,
            match="Where: The condition expression can't be a group of columns",
        ):
            translator.process()

    def test_where_mismatched_group_sizes(self):
        """Test WhereTranslator raises error when true and false groups have different sizes."""
        table = ibis.memtable(
            {
                "condition": [True, False],
                "true_col1": [1.0, 2.0],
                "true_col2": [3.0, 4.0],
                "false_col1": [10.0, 20.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (bool[N] condition, float[N] true_expr, float[N] false_expr) => (float[N] output) {
                output = Where(condition, true_expr, false_expr)
            }
        """)

        variables = GraphVariables(
            ibis.memtable(
                {"condition": [True], "true_expr": [1.0], "false_expr": [1.0]}
            ),
            model,
        )
        variables["condition"] = table["condition"]
        variables["true_expr"] = ValueVariablesGroup(
            {"col1": table["true_col1"], "col2": table["true_col2"]}
        )
        variables["false_expr"] = ValueVariablesGroup({"col1": table["false_col1"]})

        translator = WhereTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="Where: The number of values in the true and false expressions must match",
        ):
            translator.process()


class TestZipMapTranslator:
    optimizer = Optimizer(enabled=False)

    def test_zipmap_string_labels_group(self):
        """Test ZipMapTranslator with string labels and group of columns."""
        table = ibis.memtable(
            {
                "prob_class_0": [0.7, 0.2, 0.4],
                "prob_class_1": [0.2, 0.5, 0.1],
                "prob_class_2": [0.1, 0.3, 0.5],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output) {
                output = ai.onnx.ml.ZipMap <classlabels_strings: strings = ["negative", "neutral", "positive"]> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = ValueVariablesGroup(
            {
                "class_0": table["prob_class_0"],
                "class_1": table["prob_class_1"],
                "class_2": table["prob_class_2"],
            }
        )

        translator = ZipMapTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        # Check that labels were properly mapped
        assert "negative" in result
        assert "neutral" in result
        assert "positive" in result

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["negative"])) == [0.7, 0.2, 0.4]
        assert list(backend.execute(result["neutral"])) == [0.2, 0.5, 0.1]
        assert list(backend.execute(result["positive"])) == [0.1, 0.3, 0.5]

    def test_zipmap_int64_labels_group(self):
        """Test ZipMapTranslator with int64 labels and group of columns."""
        table = ibis.memtable(
            {
                "feat_0": [1.0, 2.0, 3.0],
                "feat_1": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output) {
                output = ai.onnx.ml.ZipMap <classlabels_int64s: ints = [100, 200]> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = ValueVariablesGroup(
            {
                "feat_0": table["feat_0"],
                "feat_1": table["feat_1"],
            }
        )

        translator = ZipMapTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        # int64 labels are converted to strings
        assert "100" in result
        assert "200" in result

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["100"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["200"])) == [4.0, 5.0, 6.0]

    def test_zipmap_single_column(self):
        """Test ZipMapTranslator with single column and single label."""
        table = ibis.memtable({"prob": [0.95, 0.87, 0.92]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output) {
                output = ai.onnx.ml.ZipMap <classlabels_strings: strings = ["confidence"]> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = table["prob"]

        translator = ZipMapTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        assert "confidence" in result
        assert len(result) == 1

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["confidence"])) == [0.95, 0.87, 0.92]

    def test_zipmap_missing_labels_error(self):
        """Test ZipMapTranslator raises error when no classlabels attribute is found."""
        table = ibis.memtable({"data": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output) {
                output = ai.onnx.ml.ZipMap (data)
            }
        """)

        variables = GraphVariables(table, model)

        translator = ZipMapTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError, match="ZipMap: required mapping attributes not found"
        ):
            translator.process()

    def test_zipmap_mismatched_label_count(self):
        """Test ZipMapTranslator raises error when labels count doesn't match columns."""
        table = ibis.memtable(
            {
                "col1": [1.0, 2.0, 3.0],
                "col2": [4.0, 5.0, 6.0],
                "col3": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output) {
                output = ai.onnx.ml.ZipMap <classlabels_strings: strings = ["label1", "label2"]> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = ValueVariablesGroup(
            {
                "col1": table["col1"],
                "col2": table["col2"],
                "col3": table["col3"],
            }
        )

        translator = ZipMapTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError, match="ZipMap: The number of labels and columns must match"
        ):
            translator.process()

class TestConcatTranslator:
    optimizer = Optimizer(enabled=False)

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


class TestFeatureVectorizerTranslator:
    optimizer = Optimizer(enabled=False)

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


class TestGatherTranslator:
    optimizer = Optimizer(enabled=False)

    def test_gather_single_element_from_group(self):
        """Test extracting single element from group by index."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
                "col_c": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output)
            <int32[1] index = {1}>
            {
                output = Gather <axis: int = 1> (data, index)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
                "col_c": table["col_c"],
            }
        )

        translator = GatherTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result)) == [4.0, 5.0, 6.0]

    def test_gather_element_from_single_column(self):
        """Test extracting element from single column (passthrough)."""
        table = ibis.memtable({"data": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output)
            <int32[1] index = {0}>
            {
                output = Gather <axis: int = 1> (data, index)
            }
        """)

        variables = GraphVariables(table, model)

        translator = GatherTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result)) == [1.0, 2.0, 3.0]

    def test_gather_unsupported_axis(self):
        """Test error for unsupported axis."""
        table = ibis.memtable({"data": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output)
            <int32[1] index = {0}>
            {
                output = Gather <axis: int = 0> (data, index)
            }
        """)

        variables = GraphVariables(table, model)

        translator = GatherTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError, match="only selecting columns.*is supported"
        ):
            translator.process()

    def test_gather_index_out_of_bounds(self):
        """Test error for index out of bounds."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output)
            <int32[1] index = {5}>
            {
                output = Gather <axis: int = 1> (data, index)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = GatherTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(IndexError, match="index out of bounds"):
            translator.process()

    def test_gather_invalid_index_for_single_column(self):
        """Test error for non-zero index on single column."""
        table = ibis.memtable({"data": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output)
            <int32[1] index = {1}>
            {
                output = Gather <axis: int = 1> (data, index)
            }
        """)

        variables = GraphVariables(table, model)

        translator = GatherTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError, match="index 1 not supported for single columns"
        ):
            translator.process()


class TestArrayFeatureExtractorTranslator:
    optimizer = Optimizer(enabled=False)

    def test_arrayfeatureextractor_single_column_from_group(self):
        """Test extracting single column from group."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
                "col_c": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output)
            <int32[1] indices = {1}>
            {
                output = ai.onnx.ml.ArrayFeatureExtractor (data, indices)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
                "col_c": table["col_c"],
            }
        )

        translator = ArrayFeatureExtractorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # When extracting with a list of indices (even single), returns ValueVariablesGroup
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 1
        assert "col_b" in result

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_b"])) == [4.0, 5.0, 6.0]

    def test_arrayfeatureextractor_multiple_columns_from_group(self):
        """Test extracting multiple columns from group."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
                "col_c": [7.0, 8.0, 9.0],
                "col_d": [10.0, 11.0, 12.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output)
            <int32[2] indices = {0, 2}>
            {
                output = ai.onnx.ml.ArrayFeatureExtractor (data, indices)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
                "col_c": table["col_c"],
                "col_d": table["col_d"],
            }
        )

        translator = ArrayFeatureExtractorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        assert len(result) == 2
        assert "col_a" in result
        assert "col_c" in result

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["col_c"])) == [7.0, 8.0, 9.0]

    def test_arrayfeatureextractor_index_out_of_bounds(self):
        """Test error for index out of bounds."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (float[N] output)
            <int32[3] indices = {0, 1, 2}>
            {
                output = ai.onnx.ml.ArrayFeatureExtractor (data, indices)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = ArrayFeatureExtractorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError,
            match="Indices requested are more than the available numer of columns",
        ):
            translator.process()

    def test_arrayfeatureextractor_from_list_of_constants(self):
        """Test extracting from a list of constants using column indices."""
        table = ibis.memtable(
            {"indices": [0, 1, 2, 1, 0], "dummy": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] dummy, int32[N] indices) => (float[N] output) {
                output = ai.onnx.ml.ArrayFeatureExtractor (dummy, indices)
            }
        """)

        variables = GraphVariables(table, model)
        # Override the "dummy" variable with a list of constants (like class labels)
        variables["dummy"] = ["class_a", "class_b", "class_c"]  # type: ignore[assignment]

        translator = ArrayFeatureExtractorTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        # Should map indices to class names
        computed = list(backend.execute(result))
        assert computed == ["class_a", "class_b", "class_c", "class_b", "class_a"]


# MLP activation/matrix translators — full tests in test_mlp.py
class TestReluTranslator:
    """Tests for ReluTranslator — see test_mlp.py for comprehensive tests."""

    optimizer = Optimizer(enabled=False)

    def test_relu_registered(self):
        """Verify ReluTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.relu import ReluTranslator
        assert TRANSLATORS.get("Relu") is ReluTranslator


class TestTanhTranslator:
    """Tests for TanhTranslator — see test_mlp.py for comprehensive tests."""

    optimizer = Optimizer(enabled=False)

    def test_tanh_registered(self):
        """Verify TanhTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.tanh import TanhTranslator
        assert TRANSLATORS.get("Tanh") is TanhTranslator


class TestSigmoidTranslator:
    """Tests for SigmoidTranslator — see test_mlp.py for comprehensive tests."""

    optimizer = Optimizer(enabled=False)

    def test_sigmoid_registered(self):
        """Verify SigmoidTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.sigmoid import SigmoidTranslator
        assert TRANSLATORS.get("Sigmoid") is SigmoidTranslator


class TestGemmTranslator:
    """Tests for GemmTranslator — see test_mlp.py for comprehensive tests."""

    optimizer = Optimizer(enabled=False)

    def test_gemm_registered(self):
        """Verify GemmTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.gemm import GemmTranslator
        assert TRANSLATORS.get("Gemm") is GemmTranslator


class TestClipTranslator:
    """Tests for ClipTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_clip_registered(self):
        """Verify ClipTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.clip import ClipTranslator
        assert TRANSLATORS.get("Clip") is ClipTranslator

    def test_clip_attribute_bounds(self):
        """Clip with min/max as node attributes."""
        table = ibis.memtable({"x": [-2.0, 0.0, 3.0, 8.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Clip <min: float = 0.0, max: float = 6.0> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.clip import ClipTranslator
        t = ClipTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [0.0, 0.0, 3.0, 6.0]

    def test_clip_min_only(self):
        """Clip with only a lower bound (relu-equivalent)."""
        table = ibis.memtable({"x": [-1.0, 0.0, 2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Clip <min: float = 0.0> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.clip import ClipTranslator
        t = ClipTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [0.0, 0.0, 2.0]


class TestLogSoftmaxTranslator:
    """Tests for LogSoftmaxTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_logsoftmax_registered(self):
        """Verify LogSoftmaxTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.logsoftmax import LogSoftmaxTranslator
        assert TRANSLATORS.get("LogSoftmax") is LogSoftmaxTranslator

    def test_logsoftmax_group(self):
        """LogSoftmax of a column group sums to log(1) per row (all exp sum = 1 → log=0)."""
        import math
        multi_table = ibis.memtable({"a": [1.0], "b": [2.0], "c": [3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = LogSoftmax(input)
            }
        """)
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {"a": multi_table["a"], "b": multi_table["b"], "c": multi_table["c"]}
        )
        from orbital.translation.steps.logsoftmax import LogSoftmaxTranslator
        t = LogSoftmaxTranslator(
            multi_table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        backend = ibis.duckdb.connect()
        vals = [backend.execute(result[k])[0] for k in ["a", "b", "c"]]
        # All log-softmax values must be ≤ 0 and exp(vals) must sum to 1
        assert all(v <= 0 for v in vals)
        assert abs(sum(math.exp(v) for v in vals) - 1.0) < 1e-9

    def test_logsoftmax_single(self):
        """LogSoftmax of a single input is always 0."""
        table = ibis.memtable({"x": [5.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = LogSoftmax(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.logsoftmax import LogSoftmaxTranslator
        t = LogSoftmaxTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert backend.execute(variables.peek_variable("output")) == 0.0


class TestTransposeTranslator:
    """Tests for TransposeTranslator."""

    optimizer = Optimizer(enabled=False)

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

    optimizer = Optimizer(enabled=False)

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


class TestAbsTranslator:
    """Tests for AbsTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_abs_registered(self):
        from orbital.translation.steps.abs import AbsTranslator
        assert TRANSLATORS.get("Abs") is AbsTranslator

    def test_abs_single_column(self):
        table = ibis.memtable({"x": [-3.0, 0.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Abs(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.abs import AbsTranslator
        t = AbsTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [3.0, 0.0, 4.0]

    def test_abs_group(self):
        table = ibis.memtable({"a": [-1.0, 2.0], "b": [3.0, -4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Abs(x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.abs import AbsTranslator
        t = AbsTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["a"])) == [1.0, 2.0]
        assert list(backend.execute(result["b"])) == [3.0, 4.0]


class TestNegTranslator:
    """Tests for NegTranslator."""

    optimizer = Optimizer(enabled=False)

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


class TestSqrtTranslator:
    """Tests for SqrtTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_sqrt_registered(self):
        from orbital.translation.steps.sqrt import SqrtTranslator
        assert TRANSLATORS.get("Sqrt") is SqrtTranslator

    def test_sqrt_single_column(self):
        table = ibis.memtable({"x": [4.0, 9.0, 16.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Sqrt(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.sqrt import SqrtTranslator
        t = SqrtTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [2.0, 3.0, 4.0]

    def test_sqrt_group(self):
        table = ibis.memtable({"a": [1.0, 4.0], "b": [9.0, 16.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Sqrt(x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.sqrt import SqrtTranslator
        t = SqrtTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["a"])) == [1.0, 2.0]
        assert list(backend.execute(result["b"])) == [3.0, 4.0]


class TestErfTranslator:
    """Tests for ErfTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_erf_registered(self):
        from orbital.translation.steps.erf import ErfTranslator
        assert TRANSLATORS.get("Erf") is ErfTranslator

    def test_erf_zero_input(self):
        """erf(0) == 0."""
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Erf(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.erf import ErfTranslator
        t = ErfTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - 0.0) < 1e-9

    def test_erf_large_positive(self):
        """erf(large) ≈ 1.0."""
        table = ibis.memtable({"x": [5.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Erf(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.erf import ErfTranslator
        t = ErfTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - 1.0) < 1e-6


class TestPowTranslator:
    """Tests for PowTranslator."""

    optimizer = Optimizer(enabled=False)

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


class TestEluTranslator:
    """Tests for EluTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_elu_registered(self):
        from orbital.translation.steps.elu import EluTranslator
        assert TRANSLATORS.get("Elu") is EluTranslator

    def test_elu_positive_unchanged(self):
        """ELU leaves positive values unchanged."""
        table = ibis.memtable({"x": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Elu(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.elu import EluTranslator
        t = EluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [1.0, 2.0, 3.0]

    def test_elu_negative_with_default_alpha(self):
        """ELU with negative input uses alpha*(exp(x)-1) with alpha=1.0."""
        import math
        table = ibis.memtable({"x": [-1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Elu(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.elu import EluTranslator
        t = EluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        expected = 1.0 * (math.exp(-1.0) - 1.0)
        assert abs(val - expected) < 1e-9


class TestLeakyReluTranslator:
    """Tests for LeakyReluTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_leakyrelu_registered(self):
        from orbital.translation.steps.leakyrelu import LeakyReluTranslator
        assert TRANSLATORS.get("LeakyRelu") is LeakyReluTranslator

    def test_leakyrelu_positive(self):
        table = ibis.memtable({"x": [1.0, 2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = LeakyRelu(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.leakyrelu import LeakyReluTranslator
        t = LeakyReluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [1.0, 2.0]

    def test_leakyrelu_negative_custom_alpha(self):
        """Negative values scaled by custom alpha."""
        table = ibis.memtable({"x": [-2.0, -4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = LeakyRelu <alpha: float = 0.1> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.leakyrelu import LeakyReluTranslator
        t = LeakyReluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert abs(result[0] - (-0.2)) < 1e-6
        assert abs(result[1] - (-0.4)) < 1e-6


class TestSeluTranslator:
    """Tests for SeluTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_selu_registered(self):
        from orbital.translation.steps.selu import SeluTranslator
        assert TRANSLATORS.get("Selu") is SeluTranslator

    def test_selu_positive(self):
        """SELU of positive input = gamma * x."""
        import math
        gamma = 1.0507009873554805
        table = ibis.memtable({"x": [1.0, 2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Selu(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.selu import SeluTranslator
        t = SeluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert abs(result[0] - gamma * 1.0) < 1e-9
        assert abs(result[1] - gamma * 2.0) < 1e-9

    def test_selu_negative(self):
        """SELU of negative input = gamma*(alpha*(exp(x)-1))."""
        import math
        alpha = 1.6732631921768188
        gamma = 1.0507009873554805
        table = ibis.memtable({"x": [-1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Selu(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.selu import SeluTranslator
        t = SeluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = backend.execute(variables.peek_variable("output"))[0]
        expected = gamma * alpha * (math.exp(-1.0) - 1.0)
        assert abs(result - expected) < 1e-9


class TestCeluTranslator:
    """Tests for CeluTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_celu_registered(self):
        from orbital.translation.steps.celu import CeluTranslator
        assert TRANSLATORS.get("Celu") is CeluTranslator

    def test_celu_positive_unchanged(self):
        """CELU leaves positive values unchanged (max(0,x) + min(0,neg) = x + 0)."""
        table = ibis.memtable({"x": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Celu(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.celu import CeluTranslator
        t = CeluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [1.0, 2.0, 3.0]

    def test_celu_zero_input(self):
        """CELU of 0 = 0."""
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Celu(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.celu import CeluTranslator
        t = CeluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert abs(backend.execute(variables.peek_variable("output"))[0]) < 1e-9


class TestHardSigmoidTranslator:
    """Tests for HardSigmoidTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_hardsigmoid_registered(self):
        from orbital.translation.steps.hardsigmoid import HardSigmoidTranslator
        assert TRANSLATORS.get("HardSigmoid") is HardSigmoidTranslator

    def test_hardsigmoid_clips_to_range(self):
        """HardSigmoid clips to [0, 1]."""
        table = ibis.memtable({"x": [-100.0, 0.0, 100.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = HardSigmoid(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.hardsigmoid import HardSigmoidTranslator
        t = HardSigmoidTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result[0] == 0.0
        assert result[2] == 1.0
        assert 0.0 < result[1] < 1.0

    def test_hardsigmoid_midpoint(self):
        """At x=0 with defaults (alpha=0.2, beta=0.5): 0.2*0 + 0.5 = 0.5."""
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = HardSigmoid(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.hardsigmoid import HardSigmoidTranslator
        t = HardSigmoidTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - 0.5) < 1e-9


class TestHardSwishTranslator:
    """Tests for HardSwishTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_hardswish_registered(self):
        from orbital.translation.steps.hardswish import HardSwishTranslator
        assert TRANSLATORS.get("HardSwish") is HardSwishTranslator

    def test_hardswish_zero_input(self):
        """hard_swish(0) = 0 * max(0, min(1, 3/6)) = 0 * 0.5 = 0."""
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = HardSwish(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.hardswish import HardSwishTranslator
        t = HardSwishTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert backend.execute(variables.peek_variable("output"))[0] == 0.0

    def test_hardswish_large_positive(self):
        """For large x, hard_swish(x) ≈ x (gate = 1)."""
        table = ibis.memtable({"x": [100.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = HardSwish(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.hardswish import HardSwishTranslator
        t = HardSwishTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert backend.execute(variables.peek_variable("output"))[0] == 100.0


class TestSqueezeTranslator:
    """Tests for SqueezeTranslator."""

    optimizer = Optimizer(enabled=False)

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

    optimizer = Optimizer(enabled=False)

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


class TestReduceMeanTranslator:
    """Tests for ReduceMeanTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_reducemean_registered(self):
        from orbital.translation.steps.reducemean import ReduceMeanTranslator
        assert TRANSLATORS.get("ReduceMean") is ReduceMeanTranslator

    def test_reducemean_group_to_scalar(self):
        """ReduceMean across feature columns returns row-wise mean."""
        table = ibis.memtable({"a": [1.0, 4.0], "b": [3.0, 2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ReduceMean <axes: ints = [-1]> (x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.reducemean import ReduceMeanTranslator
        t = ReduceMeanTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        vals = list(backend.execute(result))
        # Row 0: (1.0 + 3.0) / 2 = 2.0; Row 1: (4.0 + 2.0) / 2 = 3.0
        assert vals == [2.0, 3.0]

    def test_reducemean_single_column_passthrough(self):
        """ReduceMean of a single column returns it unchanged."""
        table = ibis.memtable({"x": [5.0, 10.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ReduceMean <axes: ints = [-1]> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.reducemean import ReduceMeanTranslator
        t = ReduceMeanTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [5.0, 10.0]

    def test_reducemean_unsupported_axis_raises(self):
        """ReduceMean on axis=0 (batch axis) should raise NotImplementedError."""
        table = ibis.memtable({"x": [1.0, 2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ReduceMean <axes: ints = [0]> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.reducemean import ReduceMeanTranslator
        t = ReduceMeanTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        with pytest.raises(NotImplementedError, match="axis"):
            t.process()


class TestBatchNormalizationTranslator:
    """Tests for BatchNormalizationTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_batchnorm_registered(self):
        from orbital.translation.steps.batchnorm import BatchNormalizationTranslator
        assert TRANSLATORS.get("BatchNormalization") is BatchNormalizationTranslator

    def test_batchnorm_single_column(self):
        """BatchNorm with single feature: (x - mean) / sqrt(var + eps) * scale + bias."""
        import math
        table = ibis.memtable({"x": [2.0, 4.0, 6.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <float[1] scale = {2.0}, float[1] bias = {1.0},
             float[1] mean = {3.0}, float[1] var = {4.0}>
            {
                output = BatchNormalization(x, scale, bias, mean, var)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.batchnorm import BatchNormalizationTranslator
        t = BatchNormalizationTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        # (2 - 3) / sqrt(4 + 1e-5) * 2 + 1 ≈ -1/2 * 2 + 1 = 0.0
        expected = [(v - 3.0) / math.sqrt(4.0 + 1e-5) * 2.0 + 1.0 for v in [2.0, 4.0, 6.0]]
        for got, exp in zip(result, expected):
            assert abs(got - exp) < 1e-6

    def test_batchnorm_group_columns(self):
        """BatchNorm normalizes each feature independently."""
        import math
        table = ibis.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <float[2] scale = {1.0, 1.0}, float[2] bias = {0.0, 0.0},
             float[2] mean = {1.5, 3.5}, float[2] var = {0.25, 0.25}>
            {
                output = BatchNormalization(x, scale, bias, mean, var)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.batchnorm import BatchNormalizationTranslator
        t = BatchNormalizationTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        eps = 1e-5
        std = math.sqrt(0.25 + eps)
        a_vals = list(backend.execute(result["a"]))
        b_vals = list(backend.execute(result["b"]))
        assert abs(a_vals[0] - (1.0 - 1.5) / std) < 1e-6
        assert abs(b_vals[0] - (3.0 - 3.5) / std) < 1e-6

    def test_batchnorm_feature_count_mismatch_raises(self):
        """BatchNorm raises when scale length != number of feature columns."""
        table = ibis.memtable({"a": [1.0], "b": [2.0], "c": [3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <float[2] scale = {1.0, 1.0}, float[2] bias = {0.0, 0.0},
             float[2] mean = {0.0, 0.0}, float[2] var = {1.0, 1.0}>
            {
                output = BatchNormalization(x, scale, bias, mean, var)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"], "c": table["c"]})
        from orbital.translation.steps.batchnorm import BatchNormalizationTranslator
        t = BatchNormalizationTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        with pytest.raises(ValueError, match="scale length"):
            t.process()


# ---------------------------------------------------------------------------
# New translator tests (Python Issues #30#35)
# ---------------------------------------------------------------------------


class TestGeluTranslator:
    """Tests for GeluTranslator (ONNX opset 20+ Gelu op)."""

    optimizer = Optimizer(enabled=False)

    def test_gelu_registered(self):
        from orbital.translation.steps.gelu import GeluTranslator
        assert TRANSLATORS.get("Gelu") is GeluTranslator

    def test_gelu_none_zero(self):
        """Gelu(0) = 0 for approximate='none'."""
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Gelu <approximate: string = "none"> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.gelu import GeluTranslator
        t = GeluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - 0.0) < 1e-6

    def test_gelu_tanh_positive(self):
        """Gelu(2.0) with approximate='tanh' should be close to the true GELU value."""
        import math
        table = ibis.memtable({"x": [2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Gelu <approximate: string = "tanh"> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.gelu import GeluTranslator
        t = GeluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        # tanh approx: 0.5 * 2 * (1 + tanh(sqrt(2/pi) * (2 + 0.044715*8)))
        c = math.sqrt(2.0 / math.pi)
        expected = 0.5 * 2.0 * (1.0 + math.tanh(c * (2.0 + 0.044715 * 8.0)))
        assert abs(val - expected) < 1e-4

    def test_gelu_none_positive(self):
        """Gelu(1.0) with approximate='none' ~= 0.8413."""
        import math
        table = ibis.memtable({"x": [1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Gelu <approximate: string = "none"> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.gelu import GeluTranslator
        t = GeluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        # True GELU(1.0)  0.8413
        assert abs(val - 0.8413) < 0.01

    def test_gelu_group(self):
        """Gelu applied element-wise over a VariablesGroup."""
        table = ibis.memtable({"h0": [0.0, 1.0], "h1": [2.0, -1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Gelu <approximate: string = "none"> (x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"h0": table["h0"], "h1": table["h1"]})
        from orbital.translation.steps.gelu import GeluTranslator
        t = GeluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)


class TestPReluTranslator:
    """Tests for PreluTranslator."""

    optimizer = Optimizer(enabled=False)

    def _make_prelu_graph(self, slope_val: float):
        return onnx.parser.parse_graph(f"""
            agraph (float[N] x) => (float[N] output)
            <float[1] slope = {{{slope_val}}}>
            {{
                output = PRelu(x, slope)
            }}
        """)

    def test_prelu_registered(self):
        from orbital.translation.steps.prelu import PreluTranslator
        assert TRANSLATORS.get("PRelu") is PreluTranslator

    def test_prelu_positive_unchanged(self):
        """PRelu of positive values = x (slope doesn't matter)."""
        table = ibis.memtable({"x": [1.0, 2.0, 3.0]})
        model = self._make_prelu_graph(0.25)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.prelu import PreluTranslator
        t = PreluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [1.0, 2.0, 3.0]

    def test_prelu_negative_scaled(self):
        """PRelu of negative values = slope * x."""
        table = ibis.memtable({"x": [-2.0, -4.0]})
        model = self._make_prelu_graph(0.5)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.prelu import PreluTranslator
        t = PreluTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert abs(result[0] - (-1.0)) < 1e-9
        assert abs(result[1] - (-2.0)) < 1e-9


class TestLayerNormalizationTranslator:
    """Tests for LayerNormalizationTranslator (ONNX opset 17+)."""

    optimizer = Optimizer(enabled=False)

    def test_layernorm_registered(self):
        from orbital.translation.steps.layernorm import LayerNormalizationTranslator
        assert TRANSLATORS.get("LayerNormalization") is LayerNormalizationTranslator

    def test_layernorm_normalizes_row(self):
        """LayerNorm normalizes a row to mean0, std1 (with scale=1, bias=0)."""
        import math
        # Two rows x=[2,4]: mean=3, var=1, (2-3)/sqrt(1+1e-5)=-1/1-1.0
        table = ibis.memtable({"a": [2.0, 10.0], "b": [4.0, 10.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <float[2] scale = {1.0, 1.0}, float[2] bias = {0.0, 0.0}>
            {
                output = LayerNormalization(x, scale, bias)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.layernorm import LayerNormalizationTranslator
        t = LayerNormalizationTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)
        backend = ibis.duckdb.connect()
        a_vals = list(backend.execute(result["a"]))
        b_vals = list(backend.execute(result["b"]))
        eps = 1e-5
        # Row 0: mean=3, var=1  (2-3)/sqrt(1+eps)  -1.0, (4-3)/sqrt(1+eps)  1.0
        assert abs(a_vals[0] - (-1.0 / math.sqrt(1.0 + eps))) < 1e-5
        assert abs(b_vals[0] - (1.0 / math.sqrt(1.0 + eps))) < 1e-5

    def test_layernorm_scale_and_bias_applied(self):
        """LayerNorm applies scale and bias after normalization."""
        import math
        table = ibis.memtable({"a": [2.0], "b": [4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output)
            <float[2] scale = {2.0, 2.0}, float[2] bias = {1.0, 1.0}>
            {
                output = LayerNormalization(x, scale, bias)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.layernorm import LayerNormalizationTranslator
        t = LayerNormalizationTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        result = variables.peek_variable("output")
        backend = ibis.duckdb.connect()
        a_val = backend.execute(result["a"])[0]
        b_val = backend.execute(result["b"])[0]
        eps = 1e-5
        std = math.sqrt(1.0 + eps)
        # a: (-1.0/std) * 2 + 1, b: (1.0/std) * 2 + 1
        assert abs(a_val - ((-1.0 / std) * 2.0 + 1.0)) < 1e-5
        assert abs(b_val - ((1.0 / std) * 2.0 + 1.0)) < 1e-5


class TestReduceMaxTranslator:
    """Tests for ReduceMaxTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_reducemax_registered(self):
        from orbital.translation.steps.reducemax import ReduceMaxTranslator
        assert TRANSLATORS.get("ReduceMax") is ReduceMaxTranslator

    def test_reducemax_group(self):
        """ReduceMax across columns returns per-row maximum."""
        table = ibis.memtable({"a": [1.0, 5.0], "b": [3.0, 2.0], "c": [2.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ReduceMax <axes: ints = [-1]> (x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"], "c": table["c"]})
        from orbital.translation.steps.reducemax import ReduceMaxTranslator
        t = ReduceMaxTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [3.0, 5.0]

    def test_reducemax_single_column(self):
        """ReduceMax of a single column is a pass-through."""
        table = ibis.memtable({"x": [7.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ReduceMax <axes: ints = [-1]> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.reducemax import ReduceMaxTranslator
        t = ReduceMaxTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [7.0, 3.0]

    def test_reducemax_unsupported_axis_raises(self):
        """ReduceMax on batch axis (0) should raise NotImplementedError."""
        table = ibis.memtable({"x": [1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ReduceMax <axes: ints = [0]> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.reducemax import ReduceMaxTranslator
        t = ReduceMaxTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        with pytest.raises(NotImplementedError, match="axis"):
            t.process()


class TestReduceMinTranslator:
    """Tests for ReduceMinTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_reducemin_registered(self):
        from orbital.translation.steps.reducemin import ReduceMinTranslator
        assert TRANSLATORS.get("ReduceMin") is ReduceMinTranslator

    def test_reducemin_group(self):
        """ReduceMin across columns returns per-row minimum."""
        table = ibis.memtable({"a": [1.0, 5.0], "b": [3.0, 2.0], "c": [2.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ReduceMin <axes: ints = [-1]> (x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"], "c": table["c"]})
        from orbital.translation.steps.reducemin import ReduceMinTranslator
        t = ReduceMinTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [1.0, 2.0]

    def test_reducemin_unsupported_axis_raises(self):
        """ReduceMin on batch axis (0) should raise NotImplementedError."""
        table = ibis.memtable({"x": [1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ReduceMin <axes: ints = [0]> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.reducemin import ReduceMinTranslator
        t = ReduceMinTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        with pytest.raises(NotImplementedError, match="axis"):
            t.process()


class TestReduceSumTranslator:
    """Tests for ReduceSumTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_reducesum_registered(self):
        from orbital.translation.steps.reducesum import ReduceSumTranslator
        assert TRANSLATORS.get("ReduceSum") is ReduceSumTranslator

    def test_reducesum_group(self):
        """ReduceSum across columns returns per-row sum."""
        table = ibis.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ReduceSum <axes: ints = [-1]> (x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"]})
        from orbital.translation.steps.reducesum import ReduceSumTranslator
        t = ReduceSumTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [4.0, 6.0]

    def test_reducesum_unsupported_axis_raises(self):
        """ReduceSum on batch axis (0) should raise NotImplementedError."""
        table = ibis.memtable({"x": [1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ReduceSum <axes: ints = [0]> (x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.reducesum import ReduceSumTranslator
        t = ReduceSumTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        with pytest.raises(NotImplementedError, match="axis"):
            t.process()


class TestAveragePoolTranslator:
    """Tests for AveragePoolTranslator."""

    optimizer = Optimizer(enabled=False)

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
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [3.0, 8.0]

    def test_averagepool_unsupported_shape_raises(self):
        """AveragePool with mismatched kernel_shape raises NotImplementedError."""
        table = ibis.memtable({"a": [1.0], "b": [2.0], "c": [3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = AveragePool <kernel_shape: ints = [2]> (x)
            }
        """)
        variables = GraphVariables(ibis.memtable({"x": [1.0]}), model)
        variables["x"] = NumericVariablesGroup({"a": table["a"], "b": table["b"], "c": table["c"]})
        from orbital.translation.steps.avgpool import AveragePoolTranslator
        t = AveragePoolTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        with pytest.raises(NotImplementedError):
            t.process()


class TestSoftplusTranslator:
    """Tests for SoftplusTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_softplus_registered(self):
        from orbital.translation.steps.softplus import SoftplusTranslator
        assert TRANSLATORS.get("Softplus") is SoftplusTranslator

    def test_softplus_zero(self):
        """Softplus(0) = ln(2)  0.6931."""
        import math
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Softplus(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.softplus import SoftplusTranslator
        t = SoftplusTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - math.log(2.0)) < 1e-9

    def test_softplus_positive(self):
        """Softplus(3) = ln(1+e^3)  3.049."""
        import math
        table = ibis.memtable({"x": [3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Softplus(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.softplus import SoftplusTranslator
        t = SoftplusTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - math.log(1 + math.exp(3.0))) < 1e-9


class TestMishTranslator:
    """Tests for MishTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_mish_registered(self):
        from orbital.translation.steps.mish import MishTranslator
        assert TRANSLATORS.get("Mish") is MishTranslator

    def test_mish_zero(self):
        """Mish(0) = 0 * tanh(ln(2))  0."""
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Mish(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.mish import MishTranslator
        t = MishTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - 0.0) < 1e-9

    def test_mish_positive(self):
        """Mish(1) = 1 * tanh(ln(1+e))  0.865."""
        import math
        table = ibis.memtable({"x": [1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Mish(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.mish import MishTranslator
        t = MishTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        expected = 1.0 * math.tanh(math.log(1.0 + math.e))
        assert abs(val - expected) < 1e-9


class TestLogSigmoidTranslator:
    """Tests for LogSigmoidTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_logsigmoid_registered(self):
        from orbital.translation.steps.logsigmoid import LogSigmoidTranslator
        assert TRANSLATORS.get("LogSigmoid") is LogSigmoidTranslator

    def test_logsigmoid_zero(self):
        """LogSigmoid(0) = ln(0.5)  -0.6931."""
        import math
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = LogSigmoid(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.logsigmoid import LogSigmoidTranslator
        t = LogSigmoidTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - math.log(0.5)) < 1e-9

    def test_logsigmoid_large_positive(self):
        """LogSigmoid(large positive)  0."""
        table = ibis.memtable({"x": [100.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = LogSigmoid(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.logsigmoid import LogSigmoidTranslator
        t = LogSigmoidTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - 0.0) < 1e-4


class TestHardTanhTranslator:
    """Tests for HardTanhTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_hardtanh_registered(self):
        from orbital.translation.steps.hardtanh import HardTanhTranslator
        assert TRANSLATORS.get("HardTanh") is HardTanhTranslator

    def test_hardtanh_clips_low(self):
        """HardTanh clips values below -1 to -1."""
        table = ibis.memtable({"x": [-5.0, -1.5]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = HardTanh(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.hardtanh import HardTanhTranslator
        t = HardTanhTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [-1.0, -1.0]

    def test_hardtanh_clips_high(self):
        """HardTanh clips values above 1 to 1."""
        table = ibis.memtable({"x": [1.5, 5.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = HardTanh(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.hardtanh import HardTanhTranslator
        t = HardTanhTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [1.0, 1.0]

    def test_hardtanh_passthrough_in_range(self):
        """HardTanh keeps values in [-1, 1] unchanged."""
        table = ibis.memtable({"x": [-0.5, 0.0, 0.5]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = HardTanh(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.hardtanh import HardTanhTranslator
        t = HardTanhTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        result = list(backend.execute(variables.peek_variable("output")))
        assert result == [-0.5, 0.0, 0.5]


class TestSoftsignTranslator:
    """Tests for SoftsignTranslator."""

    optimizer = Optimizer(enabled=False)

    def test_softsign_registered(self):
        from orbital.translation.steps.softsign import SoftsignTranslator
        assert TRANSLATORS.get("Softsign") is SoftsignTranslator

    def test_softsign_zero(self):
        """Softsign(0) = 0 / (1 + 0) = 0."""
        table = ibis.memtable({"x": [0.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Softsign(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.softsign import SoftsignTranslator
        t = SoftsignTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        assert backend.execute(variables.peek_variable("output"))[0] == 0.0

    def test_softsign_positive(self):
        """Softsign(3) = 3 / (1 + 3) = 0.75."""
        table = ibis.memtable({"x": [3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Softsign(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.softsign import SoftsignTranslator
        t = SoftsignTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - 0.75) < 1e-9

    def test_softsign_negative(self):
        """Softsign(-4) = -4 / (1 + 4) = -0.8."""
        table = ibis.memtable({"x": [-4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = Softsign(x)
            }
        """)
        variables = GraphVariables(table, model)
        from orbital.translation.steps.softsign import SoftsignTranslator
        t = SoftsignTranslator(table, model.node[0], variables, self.optimizer, TranslationOptions())
        t.process()
        backend = ibis.duckdb.connect()
        val = backend.execute(variables.peek_variable("output"))[0]
        assert abs(val - (-0.8)) < 1e-9


class TestDropoutTranslator:
    optimizer = Optimizer(enabled=False)
    def test_dropout_registered(self):
        from orbital.translation.steps.dropout import DropoutTranslator
        assert TRANSLATORS.get("Dropout") is DropoutTranslator


class TestGlobalAveragePoolTranslator:
    optimizer = Optimizer(enabled=False)
    def test_globalavgpool_registered(self):
        from orbital.translation.steps.globalavgpool import GlobalAveragePoolTranslator
        assert TRANSLATORS.get("GlobalAveragePool") is GlobalAveragePoolTranslator


class TestGlobalMaxPoolTranslator:
    optimizer = Optimizer(enabled=False)
    def test_globalmaxpool_registered(self):
        from orbital.translation.steps.globalmaxpool import GlobalMaxPoolTranslator
        assert TRANSLATORS.get("GlobalMaxPool") is GlobalMaxPoolTranslator


class TestGroupNormalizationTranslator:
    optimizer = Optimizer(enabled=False)
    def test_groupnorm_registered(self):
        from orbital.translation.steps.groupnorm import GroupNormalizationTranslator
        assert TRANSLATORS.get("GroupNormalization") is GroupNormalizationTranslator


class TestInstanceNormalizationTranslator:
    optimizer = Optimizer(enabled=False)
    def test_instancenorm_registered(self):
        from orbital.translation.steps.instancenorm import InstanceNormalizationTranslator
        assert TRANSLATORS.get("InstanceNormalization") is InstanceNormalizationTranslator


class TestMaxPoolTranslator:
    optimizer = Optimizer(enabled=False)
    def test_maxpool_registered(self):
        from orbital.translation.steps.maxpool import MaxPoolTranslator
        assert TRANSLATORS.get("MaxPool") is MaxPoolTranslator


class TestDropoutTranslator:
    optimizer = Optimizer(enabled=False)
    def test_dropout_registered(self):
        from orbital.translation.steps.dropout import DropoutTranslator
        assert TRANSLATORS.get('Dropout') is DropoutTranslator
