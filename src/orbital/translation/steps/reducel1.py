"""Implementation of the ReduceL1 operator.

For the SQL/tabular context, only reduction over the feature axis
(axis=-1 or axis=1) is supported, collapsing a ``VariablesGroup`` of N
columns to a single per-row L1 norm (sum of absolute values).

References
----------
https://onnx.ai/onnx/operators/onnx__ReduceL1.html
"""

import ibis

from ._base_reduce import _ReduceAxisTranslator
from ..variables import NumericVariablesGroup, VariablesGroup


class ReduceL1Translator(_ReduceAxisTranslator):
    """Translate the ONNX ReduceL1 operator: sum of absolute values."""

    def process(self) -> None:
        """Translate the ReduceL1 node, writing result to the graph."""
        args = self._extract_reduce_args("ReduceL1")
        if args is None:
            return
        data, keepdims = args
        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            cols = list(data.values())
            if not cols:
                raise ValueError("ReduceL1: input group must have at least one column.")
            sum_expr: ibis.expr.types.NumericValue = cols[0].abs()
            for col in cols[1:]:
                sum_expr = sum_expr + col.abs()
            agg = self._optimizer.fold_operation(sum_expr)
            if keepdims == 1:
                from ..variables import ValueVariablesGroup
                self.set_output(ValueVariablesGroup({"out_0": agg}))
            else:
                self.set_output(agg)
        else:
            # Single column: L1 norm is its absolute value.
            self.set_output(self._optimizer.fold_operation(data.abs()))
