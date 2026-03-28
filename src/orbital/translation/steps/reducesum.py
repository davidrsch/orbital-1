"""Implementation of the ReduceSum operator.

For the SQL/tabular context, only reduction over the feature axis
(axis=-1 or axis=1) is supported, collapsing a ``VariablesGroup`` of N
columns to a single per-row sum.

References
----------
https://onnx.ai/onnx/operators/onnx__ReduceSum.html
"""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class ReduceSumTranslator(Translator):
    """Translate the ONNX ReduceSum operator over the feature axis."""

    def process(self) -> None:
        """Translate the ReduceSum node, writing result to the graph."""
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
                f"ReduceSum: only axis=-1 or axis=1 (feature axis) is supported, got {axes}"
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
            if len(cols) == 0:
                raise ValueError(
                    "ReduceSum: input group must have at least one column."
                )
            sum_expr: ibis.expr.types.NumericValue = cols[0]
            for col in cols[1:]:
                sum_expr = sum_expr + col
            result = self._optimizer.fold_operation(sum_expr)
            if keepdims == 1:
                from ..variables import ValueVariablesGroup
                self.set_output(ValueVariablesGroup({"out_0": result}))
            else:
                self.set_output(result)
        else:
            # Single column: sum is itself.
            self.set_output(data)
