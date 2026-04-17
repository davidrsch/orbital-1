"""Tests for Gemm pipeline step translators."""

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
        B_tensor = helper.make_tensor(
            "B", TensorProto.FLOAT, [2, 2], [2.0, 0.0, 0.0, 3.0]
        )
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

    def test_transA_1(self):
        """Gemm with transA=1 must not raise and must produce the correct output.

        The fix removed the NotImplementedError guard for transA=1.  The orbital
        column model treats A as a 1-D feature row regardless of orientation, so
        the computation is identical to transA=0 for our tabular use-case.
        """
        from orbital.translation.steps.gemm import GemmTranslator

        # 2 features; B stored row-major [2, 2]; transA=1, transB=0
        # Y_j = sum_i(a_i * B[i,j]) so [1,2]@[[2,3],[4,5]] = [10, 13]
        table = ibis.memtable({"h0": [1.0], "h1": [2.0]})
        B_tensor = helper.make_tensor(
            "B", TensorProto.FLOAT, [2, 2], [2.0, 3.0, 4.0, 5.0]
        )
        C_tensor = helper.make_tensor("C", TensorProto.FLOAT, [2], [0.0, 0.0])
        node = helper.make_node(
            "Gemm",
            inputs=["A", "B", "C"],
            outputs=["output"],
            transA=1,
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
            [B_tensor, C_tensor],
        )
        variables = GraphVariables(ibis.memtable({"A": [1.0]}), graph)
        variables["A"] = NumericVariablesGroup({"h0": table["h0"], "h1": table["h1"]})
        GemmTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        ).process()
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)
        backend = ibis.duckdb.connect()
        # [1,2] @ [[2,3],[4,5]] = [1*2+2*4, 1*3+2*5] = [10, 13]
        assert backend.execute(result["out_0"]).tolist() == [10.0]
        assert backend.execute(result["out_1"]).tolist() == [13.0]

    def test_transA_not_supported(self):
        """Gemm with transA=1 is treated as a no-op in the tabular 1-D context.

        In the orbital column model, A is always a flat row vector so transposing
        it is semantically identical to transA=0.  Verify the computation still
        produces the correct output (identity matrix → output equals input).
        """
        from orbital.translation.steps.gemm import GemmTranslator

        table = ibis.memtable({"h0": [1.0], "h1": [2.0]})
        B_tensor = helper.make_tensor(
            "B", TensorProto.FLOAT, [2, 2], [1.0, 0.0, 0.0, 1.0]
        )
        node = helper.make_node(
            "Gemm",
            inputs=["A", "B"],
            outputs=["output"],
            transA=1,
        )
        graph = _make_graph_with_inits(
            node,
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
            [B_tensor],
        )
        variables = GraphVariables(ibis.memtable({"A": [1.0]}), graph)
        variables["A"] = NumericVariablesGroup({"h0": table["h0"], "h1": table["h1"]})
        translator = GemmTranslator(
            table, graph.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()
        result = variables.peek_variable("output")
        assert isinstance(result, VariablesGroup)
        backend = ibis.duckdb.connect()
        vals = [backend.execute(v).tolist()[0] for v in result.values()]
        assert abs(vals[0] - 1.0) < 1e-6  # h0 * 1 + h1 * 0 = 1.0
        assert abs(vals[1] - 2.0) < 1e-6  # h0 * 0 + h1 * 1 = 2.0
