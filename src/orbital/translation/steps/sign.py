"""Implementation of the Sign operator."""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class SignTranslator(Translator):
    """Translate the ONNX Sign operator: element-wise sign function.

    Returns:

    * ``+1`` when ``x > 0``
    * ``0`` when ``x == 0``
    * ``-1`` when ``x < 0``

    References
    ----------
    https://onnx.ai/onnx/operators/onnx__Sign.html
    """

    def process(self) -> None:
        """Translate the Sign node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])

        def _sign(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
            return ibis.cases(
                (v > ibis.literal(0.0), ibis.literal(1.0)),
                (v == ibis.literal(0.0), ibis.literal(0.0)),
                else_=ibis.literal(-1.0),
            )

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup(
                {k: self._optimizer.fold_operation(_sign(v)) for k, v in data.items()}
            )
        else:
            result = self._optimizer.fold_operation(_sign(data))

        self.set_output(result)
