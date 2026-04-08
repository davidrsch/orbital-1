"""Implementation of the ReduceProd operator.

For the SQL/tabular context, only reduction over the feature axis
(axis=-1 or axis=1) is supported, collapsing a ``VariablesGroup`` of N
columns to a single per-row product.

References
----------
https://onnx.ai/onnx/operators/onnx__ReduceProd.html
"""

import ibis

from ._base_reduce import _ReduceAxisTranslator
from ..variables import NumericVariablesGroup, VariablesGroup


class ReduceProdTranslator(_ReduceAxisTranslator):
    """Translate the ONNX ReduceProd operator over the feature axis."""

    def process(self) -> None:
        """Translate the ReduceProd node, writing result to the graph."""
        args = self._extract_reduce_args("ReduceProd")
        if args is None:
            return
        data, keepdims = args
        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            cols = list(data.values())
            if len(cols) == 0:
                raise ValueError(
                    "ReduceProd: input group must have at least one column."
                )
            prod_expr: ibis.expr.types.NumericValue = cols[0]
            for col in cols[1:]:
                prod_expr = prod_expr * col
            agg = self._optimizer.fold_operation(prod_expr)
            if keepdims == 1:
                from ..variables import ValueVariablesGroup
                self.set_output(ValueVariablesGroup({"out_0": agg}))
            else:
                self.set_output(agg)
        else:
            # Single column: product is itself.
            self.set_output(data)
