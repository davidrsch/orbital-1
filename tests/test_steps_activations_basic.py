"""Tests for activations pipeline step translators."""

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

class TestSoftmaxTranslator:

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


# MLP activation/matrix translators — full tests in test_mlp.py

class TestReluTranslator:
    """Tests for ReluTranslator — see test_mlp.py for comprehensive tests."""


    def test_relu_registered(self):
        """Verify ReluTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.relu import ReluTranslator
        assert TRANSLATORS.get("Relu") is ReluTranslator

    def test_relu_correctness(self):
        """Relu clamps negative values to 0 and passes positive values through."""
        from orbital.translation.steps.relu import ReluTranslator
        table = ibis.memtable({"input": [-2.0, 0.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Relu(input)
            }
        """)
        variables = GraphVariables(table, model)
        ReluTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = ibis.duckdb.connect().execute(
            variables.peek_variable("output")
        ).tolist()
        assert result == [0.0, 0.0, 3.0]

    def test_relu_non_numeric_raises(self):
        """Relu raises ValueError when the input variable is non-numeric."""
        import pytest
        from orbital.translation.steps.relu import ReluTranslator
        table = ibis.memtable({"input": [1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Relu(input)
            }
        """)
        variables = GraphVariables(table, model)
        # Override the "input" variable with a non-numeric (string) column.
        str_table = ibis.memtable({"val": ["a", "b"]})
        variables["input"] = str_table["val"]
        with pytest.raises(ValueError, match="numeric"):
            ReluTranslator(
                table, model.node[0], variables, self.optimizer, TranslationOptions()
            ).process()



class TestTanhTranslator:
    """Tests for TanhTranslator — see test_mlp.py for comprehensive tests."""


    def test_tanh_registered(self):
        """Verify TanhTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.tanh import TanhTranslator
        assert TRANSLATORS.get("Tanh") is TanhTranslator

    def test_tanh_correctness(self):
        """tanh(0) = 0; tanh preserves sign."""
        import math
        from orbital.translation.steps.tanh import TanhTranslator
        table = ibis.memtable({"input": [0.0, 1.0, -1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Tanh(input)
            }
        """)
        variables = GraphVariables(table, model)
        TanhTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = ibis.duckdb.connect().execute(
            variables.peek_variable("output")
        ).tolist()
        assert result[0] == 0.0
        assert abs(result[1] - math.tanh(1.0)) < 1e-9
        assert abs(result[2] - math.tanh(-1.0)) < 1e-9

    def test_tanh_non_numeric_raises(self):
        """Tanh raises ValueError when the input variable is non-numeric."""
        import pytest
        from orbital.translation.steps.tanh import TanhTranslator
        table = ibis.memtable({"input": [1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Tanh(input)
            }
        """)
        variables = GraphVariables(table, model)
        str_table = ibis.memtable({"val": ["a", "b"]})
        variables["input"] = str_table["val"]
        with pytest.raises(ValueError, match="numeric"):
            TanhTranslator(
                table, model.node[0], variables, self.optimizer, TranslationOptions()
            ).process()



class TestSigmoidTranslator:
    """Tests for SigmoidTranslator — see test_mlp.py for comprehensive tests."""


    def test_sigmoid_registered(self):
        """Verify SigmoidTranslator is registered in TRANSLATORS."""
        from orbital.translation.steps.sigmoid import SigmoidTranslator
        assert TRANSLATORS.get("Sigmoid") is SigmoidTranslator

    def test_sigmoid_correctness(self):
        """sigmoid(0) = 0.5; sigmoid is monotonically increasing."""
        from orbital.translation.steps.sigmoid import SigmoidTranslator
        table = ibis.memtable({"input": [0.0, 1.0, -1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Sigmoid(input)
            }
        """)
        variables = GraphVariables(table, model)
        SigmoidTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = ibis.duckdb.connect().execute(
            variables.peek_variable("output")
        ).tolist()
        assert abs(result[0] - 0.5) < 1e-9
        assert result[1] > 0.5  # sigmoid(positive) > 0.5
        assert result[2] < 0.5  # sigmoid(negative) < 0.5

    def test_sigmoid_non_numeric_raises(self):
        """Sigmoid raises ValueError when the input variable is non-numeric."""
        import pytest
        from orbital.translation.steps.sigmoid import SigmoidTranslator
        table = ibis.memtable({"input": [1.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Sigmoid(input)
            }
        """)
        variables = GraphVariables(table, model)
        str_table = ibis.memtable({"val": ["a", "b"]})
        variables["input"] = str_table["val"]
        with pytest.raises(ValueError, match="numeric"):
            SigmoidTranslator(
                table, model.node[0], variables, self.optimizer, TranslationOptions()
            ).process()



class TestLeakyReluTranslator:
    """Tests for LeakyReluTranslator."""


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



class TestHardSigmoidTranslator:
    """Tests for HardSigmoidTranslator."""


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



class TestSoftsignTranslator:
    """Tests for SoftsignTranslator."""


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



class TestLogSoftmaxTranslator:
    """Tests for LogSoftmaxTranslator."""


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


# ---------------------------------------------------------------------------
# Unit tests: SwishTranslator
# ---------------------------------------------------------------------------


