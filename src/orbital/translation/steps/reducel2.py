"""Implementation of the ReduceL2 operator.

For the SQL/tabular context, only reduction over the feature axis
(axis=-1 or axis=1) is supported, collapsing a ``VariablesGroup`` of N
columns to a single per-row L2 norm (Euclidean norm).

References
----------
https://onnx.ai/onnx/operators/onnx__ReduceL2.html
"""

import ibis

from ._base_reduce import _ReduceAxisTranslator
from ..variables import NumericVariablesGroup, VariablesGroup


class ReduceL2Translator(_ReduceAxisTranslator):
    """Translate the ONNX ReduceL2 operator: sqrt(sum of squares)."""

    def process(self) -> None:
        """Translate the ReduceL2 node, writing result to the graph."""
        args = self._extract_reduce_args("ReduceL2")
        if args is None:
            return
        data, keepdims = args
        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            cols = list(data.values())
            if not cols:
                raise ValueError("ReduceL2: input group must have at least one column.")
            sum_sq: ibis.expr.types.NumericValue = cols[0] * cols[0]
            for col in cols[1:]:
                sum_sq = sum_sq + (col * col)
            agg = self._optimizer.fold_operation(sum_sq.sqrt())
            if keepdims == 1:
                from ..variables import ValueVariablesGroup
                self.set_output(ValueVariablesGroup({"out_0": agg}))
            else:
                self.set_output(agg)
        else:
            # Single column: L2 norm is its absolute value.
            self.set_output(self._optimizer.fold_operation((data * data).sqrt()))
