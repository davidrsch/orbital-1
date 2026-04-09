"""Implementation of the ReduceMax operator.

For the SQL/tabular context, only reduction over the feature axis
(axis=-1 or axis=1) is supported, collapsing a ``VariablesGroup`` of N
columns to a single per-row maximum.

References
----------
https://onnx.ai/onnx/operators/onnx__ReduceMax.html
"""

import ibis

from ._base_reduce import _ReduceAxisTranslator
from ..variables import NumericVariablesGroup, VariablesGroup


class ReduceMaxTranslator(_ReduceAxisTranslator):
    """Translate the ONNX ReduceMax operator over the feature axis."""

    def process(self) -> None:
        """Translate the ReduceMax node, writing result to the graph."""
        args = self._extract_reduce_args("ReduceMax")
        if args is None:
            return
        data, keepdims = args
        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            cols = list(data.values())
            if len(cols) == 0:
                raise ValueError(
                    "ReduceMax: input group must have at least one column."
                )
            max_expr: ibis.expr.types.NumericValue = cols[0]
            for col in cols[1:]:
                max_expr = ibis.greatest(max_expr, col)
            agg = self._optimizer.fold_operation(max_expr)
            self._set_reduce_output(agg, keepdims)
        else:
            # Single column: max of one element is itself.
            self.set_output(data)
