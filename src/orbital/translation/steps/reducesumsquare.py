"""Implementation of the ReduceSumSquare operator (ONNX opset 1+).

Computes the sum of squares of elements along the specified axes:

    y = x_0^2 + x_1^2 + ... + x_{n-1}^2

Only reduction over the feature axis (axis=-1 or axis=1) is supported.

References
----------
https://onnx.ai/onnx/operators/onnx__ReduceSumSquare.html
"""

from ..variables import NumericVariablesGroup, VariablesGroup
from ._base_reduce import _ReduceAxisTranslator


class ReduceSumSquareTranslator(_ReduceAxisTranslator):
    """Translate the ONNX ReduceSumSquare operator: sum of squares."""

    def process(self) -> None:
        """Translate the ReduceSumSquare node, writing result to the graph."""
        args = self._extract_reduce_args("ReduceSumSquare")
        if args is None:
            return
        data, keepdims = args
        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            cols = list(data.values())
            if not cols:
                raise ValueError(
                    "ReduceSumSquare: input group must have at least one column."
                )
            sum_sq = cols[0] ** 2
            for col in cols[1:]:
                sum_sq = sum_sq + col**2
            agg = self._optimizer.fold_operation(sum_sq)
            self._set_reduce_output(agg, keepdims)
        else:
            self.set_output(self._optimizer.fold_operation(data**2))
