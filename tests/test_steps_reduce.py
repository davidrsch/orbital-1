"""Tests for reduce pipeline step translators."""

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


class TestReduceMeanTranslator:
    """Tests for ReduceMeanTranslator."""


    def test_reducemean_registered(self):
        from orbital.translation.steps.reducemean import ReduceMeanTranslator
        assert TRANSLATORS.get("ReduceMean") is ReduceMeanTranslator

    def test_reducemean_group_to_scalar(self):
        """ReduceMean across feature columns returns row-wise mean (keepdims=0)."""
        table = ibis.memtable({"a": [1.0, 4.0], "b": [3.0, 2.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ReduceMean <axes: ints = [-1], keepdims: int = 0> (x)
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


class TestReduceMaxTranslator:
    """Tests for ReduceMaxTranslator."""


    def test_reducemax_registered(self):
        from orbital.translation.steps.reducemax import ReduceMaxTranslator
        assert TRANSLATORS.get("ReduceMax") is ReduceMaxTranslator

    def test_reducemax_group(self):
        """ReduceMax across columns returns per-row maximum (keepdims=0)."""
        table = ibis.memtable({"a": [1.0, 5.0], "b": [3.0, 2.0], "c": [2.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ReduceMax <axes: ints = [-1], keepdims: int = 0> (x)
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


    def test_reducemin_registered(self):
        from orbital.translation.steps.reducemin import ReduceMinTranslator
        assert TRANSLATORS.get("ReduceMin") is ReduceMinTranslator

    def test_reducemin_group(self):
        """ReduceMin across columns returns per-row minimum (keepdims=0)."""
        table = ibis.memtable({"a": [1.0, 5.0], "b": [3.0, 2.0], "c": [2.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ReduceMin <axes: ints = [-1], keepdims: int = 0> (x)
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


    def test_reducesum_registered(self):
        from orbital.translation.steps.reducesum import ReduceSumTranslator
        assert TRANSLATORS.get("ReduceSum") is ReduceSumTranslator

    def test_reducesum_group(self):
        """ReduceSum across columns returns per-row sum (keepdims=0)."""
        table = ibis.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] x) => (float[N] output) {
                output = ReduceSum <axes: ints = [-1], keepdims: int = 0> (x)
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

