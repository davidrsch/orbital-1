"""Implementation of the HardSwish operator."""
import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class HardSwishTranslator(Translator):
    def process(self) -> None:
        # https://onnx.ai/onnx/operators/onnx__HardSwish.html
        # hard_swish(x) = x * max(0, min(1, (x + 3) / 6))
        data = self._variables.consume(self.inputs[0])

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("HardSwish: The input must be a numeric column.")

        def _hard_swish(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
            gate = ibis.greatest(
                ibis.literal(0.0),
                ibis.least(
                    ibis.literal(1.0),
                    (v + ibis.literal(3.0)) / ibis.literal(6.0),
                ),
            )
            return v * gate

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup({
                k: self._optimizer.fold_operation(_hard_swish(v))
                for k, v in data.items()
            })
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(_hard_swish(data))

        self.set_output(result)
