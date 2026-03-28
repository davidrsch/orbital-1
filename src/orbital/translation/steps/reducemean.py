"""Implementation of the ReduceMean operator."""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


class ReduceMeanTranslator(Translator):
    """Translate the ONNX ReduceMean operator over the feature axis."""

    def process(self) -> None:
        """Translate the ReduceMean node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__ReduceMean.html
        # Only axis=-1 / axis=1 (reduce over the feature dimension) is supported.
        data = self._variables.consume(self.inputs[0])
        keepdims = int(self._attributes.get("keepdims", 1))

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
                f"ReduceMean: only axis=-1 or axis=1 (feature axis) is supported, got {axes}"
            )

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            cols = list(data.values())
            n = len(cols)
            if n == 0:
                raise ValueError(
                    "ReduceMean: input group must have at least one column."
                )
            mean_expr: ibis.expr.types.NumericValue = cols[0]
            for col in cols[1:]:
                mean_expr = mean_expr + col
            mean_expr = mean_expr / ibis.literal(float(n))
            result = self._optimizer.fold_operation(mean_expr)
            if keepdims == 1:
                self.set_output(ValueVariablesGroup({"out_0": result}))
            else:
                self.set_output(result)
        else:
            # Single column: mean of a scalar column is itself.
            self.set_output(data)
