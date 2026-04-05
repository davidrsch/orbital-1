"""Implementation of the Shrink operator (ONNX opset 9).

Element-wise soft/hard shrinkage:

    if x < -lambd:  y = x + bias
    if x > +lambd:  y = x - bias
    otherwise:      y = 0

Attributes
----------
lambd : float, default 0.5
bias  : float, default 0.0

References
----------
https://onnx.ai/onnx/operators/onnx__Shrink.html
"""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class ShrinkTranslator(Translator):
    """Translate the ONNX Shrink operator."""

    def process(self) -> None:
        """Translate the Shrink node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])
        lambd = float(self._attributes.get("lambd", 0.5))
        bias = float(self._attributes.get("bias", 0.0))

        def _shrink(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
            neg = ibis.literal(-lambd)
            pos = ibis.literal(lambd)
            b = ibis.literal(bias)
            return ibis.cases(
                (v < neg, v + b),
                (v > pos, v - b),
                else_=ibis.literal(0.0),
            )

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup(
                {k: self._optimizer.fold_operation(_shrink(v)) for k, v in data.items()}
            )
        else:
            result = self._optimizer.fold_operation(_shrink(data))

        self.set_output(result)
