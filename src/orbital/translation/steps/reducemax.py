"""Implementation of the ReduceMax operator.

For the SQL/tabular context, only reduction over the feature axis
(axis=-1 or axis=1) is supported, collapsing a ``VariablesGroup`` of N
columns to a single per-row maximum.

References
----------
https://onnx.ai/onnx/operators/onnx__ReduceMax.html
"""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class ReduceMaxTranslator(Translator):
    """Translate the ONNX ReduceMax operator over the feature axis."""

    def process(self) -> None:
        """Translate the ReduceMax node, writing result to the graph."""
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
                f"ReduceMax: only axis=-1 or axis=1 (feature axis) is supported, got {axes}"
            )

        keepdims = int(self._attributes.get("keepdims", 1))

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
            result = self._optimizer.fold_operation(max_expr)
            if keepdims == 1:
                from ..variables import ValueVariablesGroup
                self.set_output(ValueVariablesGroup({"out_0": result}))
            else:
                self.set_output(result)
        else:
            # Single column: max of one element is itself.
            self.set_output(data)
