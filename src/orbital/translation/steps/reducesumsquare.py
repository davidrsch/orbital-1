"""Implementation of the ReduceSumSquare operator (ONNX opset 1+).

Computes the sum of squares of elements along the specified axes:

    y = x_0^2 + x_1^2 + ... + x_{n-1}^2

Only reduction over the feature axis (axis=-1 or axis=1) is supported.

References
----------
https://onnx.ai/onnx/operators/onnx__ReduceSumSquare.html
"""

from ..translator import Translator
from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


class ReduceSumSquareTranslator(Translator):
    """Translate the ONNX ReduceSumSquare operator: sum of squares."""

    def process(self) -> None:
        """Translate the ReduceSumSquare node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])

        axes = self._attributes.get("axes", None)
        if axes is not None:
            axes = list(axes)

        if len(self.inputs) > 1:
            axes_val = self._variables.get_initializer_value(self.inputs[1])
            if axes_val is not None:
                axes = [int(a) for a in axes_val]

        if axes is not None and not all(a in (-1, 1) for a in axes):
            raise NotImplementedError(
                f"ReduceSumSquare: only axis=-1 or axis=1 (feature axis) is supported, got {axes}"
            )

        keepdims = int(self._attributes.get("keepdims", 1))
        noop_with_empty_axes = int(self._attributes.get("noop_with_empty_axes", 0))

        if noop_with_empty_axes == 1 and (axes is None or len(axes) == 0):
            self.set_output(data)
            return

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            cols = list(data.values())
            if not cols:
                raise ValueError(
                    "ReduceSumSquare: input group must have at least one column."
                )
            sum_sq = cols[0] ** 2
            for col in cols[1:]:
                sum_sq = sum_sq + col ** 2
            result = self._optimizer.fold_operation(sum_sq)
            if keepdims == 1:
                self.set_output(ValueVariablesGroup({"out_0": result}))
            else:
                self.set_output(result)
        else:
            self.set_output(self._optimizer.fold_operation(data ** 2))
