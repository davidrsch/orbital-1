"""Tests for pooling and dropout pipeline step translators (split from test_steps_convtranspose.py)."""

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
        result = (
            ibis.duckdb.connect().execute(variables.peek_variable("output")).tolist()
        )
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
        result = (
            ibis.duckdb.connect().execute(variables.peek_variable("output")).tolist()
        )
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

        t = AveragePoolTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        t.process()
        backend = ibis.duckdb.connect()
        assert list(backend.execute(variables.peek_variable("output"))) == [
            1.0,
            2.0,
            3.0,
        ]

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

        t = AveragePoolTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
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

        t = AveragePoolTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
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
        result = ibis.duckdb.connect().execute(list(out.values())[0]).tolist()
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
        result = (
            ibis.duckdb.connect().execute(variables.peek_variable("output")).tolist()
        )
        assert result == [1.0, 2.0, 3.0]
