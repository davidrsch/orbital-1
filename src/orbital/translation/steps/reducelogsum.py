"""Implementation of the ReduceLogSum operator (ONNX opset 1+).

Computes the log of the sum of elements along the specified axes:

    y = log(x_0 + x_1 + ... + x_{n-1})

Only reduction over the feature axis (axis=-1 or axis=1) is supported.

References
----------
https://onnx.ai/onnx/operators/onnx__ReduceLogSum.html
"""

from ..variables import NumericVariablesGroup, VariablesGroup
from ._base_reduce import _ReduceAxisTranslator


class ReduceLogSumTranslator(_ReduceAxisTranslator):
    """Translate the ONNX ReduceLogSum operator: log(sum of elements)."""

    def process(self) -> None:
        """Translate the ReduceLogSum node, writing result to the graph."""
        args = self._extract_reduce_args("ReduceLogSum")
        if args is None:
            return
        data, keepdims = args
        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            cols = list(data.values())
            if not cols:
                raise ValueError(
                    "ReduceLogSum: input group must have at least one column."
                )
            sum_expr = cols[0]
            for col in cols[1:]:
                sum_expr = sum_expr + col
            agg = self._optimizer.fold_operation(sum_expr.ln())
            self._set_reduce_output(agg, keepdims)
        else:
            self.set_output(self._optimizer.fold_operation(data.ln()))
