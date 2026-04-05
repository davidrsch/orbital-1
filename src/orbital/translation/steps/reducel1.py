"""Implementation of the ReduceL1 operator.

For the SQL/tabular context, only reduction over the feature axis
(axis=-1 or axis=1) is supported, collapsing a ``VariablesGroup`` of N
columns to a single per-row L1 norm (sum of absolute values).

References
----------
https://onnx.ai/onnx/operators/onnx__ReduceL1.html
"""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class ReduceL1Translator(Translator):
    """Translate the ONNX ReduceL1 operator: sum of absolute values."""

    def process(self) -> None:
        """Translate the ReduceL1 node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])

        # Axes can be an attribute (opset < 18) or a second input (opset >= 18).
        axes = self._attributes.get("axes", None)
        if axes is not None:
            axes = list(axes)

        if len(self.inputs) > 1:
            axes_val = self._variables.get_initializer_value(self.inputs[1])
            if axes_val is not None:
                axes = [int(a) for a in axes_val]

        if axes is not None and not all(a in (-1, 1) for a in axes):
            raise NotImplementedError(
                f"ReduceL1: only axis=-1 or axis=1 (feature axis) is supported, got {axes}"
            )

        keepdims = int(self._attributes.get("keepdims", 1))
        noop_with_empty_axes = int(self._attributes.get("noop_with_empty_axes", 0))

        # opset 18+: when axes input is empty and noop_with_empty_axes=1, pass through.
        if noop_with_empty_axes == 1 and (axes is None or len(axes) == 0):
            self.set_output(data)
            return

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            cols = list(data.values())
            if not cols:
                raise ValueError("ReduceL1: input group must have at least one column.")
            sum_expr: ibis.expr.types.NumericValue = cols[0].abs()
            for col in cols[1:]:
                sum_expr = sum_expr + col.abs()
            result = self._optimizer.fold_operation(sum_expr)
            if keepdims == 1:
                from ..variables import ValueVariablesGroup
                self.set_output(ValueVariablesGroup({"out_0": result}))
            else:
                self.set_output(result)
        else:
            # Single column: L1 norm is its absolute value.
            self.set_output(self._optimizer.fold_operation(data.abs()))
