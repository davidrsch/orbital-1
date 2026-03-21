"""Implementation of the Sqrt operator."""
import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class SqrtTranslator(Translator):
    def process(self) -> None:
        # https://onnx.ai/onnx/operators/onnx__Sqrt.html
        data = self._variables.consume(self.inputs[0])

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("Sqrt: The input must be a numeric column.")

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup({
                k: self._optimizer.fold_operation(v.sqrt())
                for k, v in data.items()
            })
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(data.sqrt())

        self.set_output(result)
