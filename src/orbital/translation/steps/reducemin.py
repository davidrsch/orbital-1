"""Implementation of the ReduceMin operator.

For the SQL/tabular context, only reduction over the feature axis
(axis=-1 or axis=1) is supported, collapsing a ``VariablesGroup`` of N
columns to a single per-row minimum.

References
----------
https://onnx.ai/onnx/operators/onnx__ReduceMin.html
"""

import ibis

from ..variables import NumericVariablesGroup, VariablesGroup
from ._base_reduce import _ReduceAxisTranslator


class ReduceMinTranslator(_ReduceAxisTranslator):
    """Translate the ONNX ReduceMin operator over the feature axis."""

    def process(self) -> None:
        """Translate the ReduceMin node, writing result to the graph."""
        args = self._extract_reduce_args("ReduceMin")
        if args is None:
            return
        data, keepdims = args
        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            cols = list(data.values())
            if len(cols) == 0:
                raise ValueError(
                    "ReduceMin: input group must have at least one column."
                )
            min_expr: ibis.expr.types.NumericValue = cols[0]
            for col in cols[1:]:
                min_expr = ibis.least(min_expr, col)
            agg = self._optimizer.fold_operation(min_expr)
            self._set_reduce_output(agg, keepdims)
        else:
            # Single column: min of one element is itself.
            self.set_output(data)
