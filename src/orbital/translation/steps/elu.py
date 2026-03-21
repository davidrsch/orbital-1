"""Implementation of the Elu operator."""
import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class EluTranslator(Translator):
    def process(self) -> None:
        # https://onnx.ai/onnx/operators/onnx__Elu.html
        # elu(x) = x if x >= 0 else alpha * (exp(x) - 1)
        data = self._variables.consume(self.inputs[0])
        alpha = float(self._attributes.get("alpha", 1.0))

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("Elu: The input must be a numeric column.")

        def _elu(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
            return ibis.cases(
                (v >= ibis.literal(0.0), v),
                else_=ibis.literal(alpha) * (v.exp() - ibis.literal(1.0)),
            )

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup({
                k: self._optimizer.fold_operation(_elu(v))
                for k, v in data.items()
            })
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(_elu(data))

        self.set_output(result)
