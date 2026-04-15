"""Implementation of the ReduceLogSumExp operator (ONNX opset 1+).

Computes the log of the sum of exponentials of elements along the specified
axes using the numerically stable max-subtraction trick:

    y = log(sum_i exp(x_i))
      = max(x) + log(sum_i exp(x_i - max(x)))

Only reduction over the feature axis (axis=-1 or axis=1) is supported.

References
----------
https://onnx.ai/onnx/operators/onnx__ReduceLogSumExp.html
"""

import ibis

from ._base_reduce import _ReduceAxisTranslator
from ..variables import NumericVariablesGroup, VariablesGroup


class ReduceLogSumExpTranslator(_ReduceAxisTranslator):
    """Translate the ONNX ReduceLogSumExp operator: log(sum(exp(x_i)))."""

    def process(self) -> None:
        """Translate the ReduceLogSumExp node, writing result to the graph."""
        args = self._extract_reduce_args("ReduceLogSumExp")
        if args is None:
            return
        data, keepdims = args
        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            cols = list(data.values())
            if not cols:
                raise ValueError(
                    "ReduceLogSumExp: input group must have at least one column."
                )
            # Numerically stable reduction: max + log(sum(exp(x - max)))
            max_expr: ibis.expr.types.NumericValue = cols[0]
            for col in cols[1:]:
                max_expr = ibis.greatest(max_expr, col)
            shifted_sum = cols[0].exp()
            # max_expr is the row-wise max; subtract it from each column before
            # exponentiating to avoid overflow.
            shifted_sum = (cols[0] - max_expr).exp()
            for col in cols[1:]:
                shifted_sum = shifted_sum + (col - max_expr).exp()
            agg = self._optimizer.fold_operation(max_expr + shifted_sum.ln())
            self._set_reduce_output(agg, keepdims)
        else:
            # Single scalar: log(exp(x)) = x
            self.set_output(data)
