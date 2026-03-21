"""Implementation of the Celu operator."""
import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class CeluTranslator(Translator):
    def process(self) -> None:
        # https://onnx.ai/onnx/operators/onnx__Celu.html
        # celu(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
        data = self._variables.consume(self.inputs[0])
        alpha = float(self._attributes.get("alpha", 1.0))

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("Celu: The input must be a numeric column.")

        def _celu(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
            pos = ibis.greatest(ibis.literal(0.0), v)
            neg = ibis.least(
                ibis.literal(0.0),
                ibis.literal(alpha) * ((v / ibis.literal(alpha)).exp() - ibis.literal(1.0)),
            )
            return pos + neg

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup({
                k: self._optimizer.fold_operation(_celu(v))
                for k, v in data.items()
            })
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(_celu(data))

        self.set_output(result)
