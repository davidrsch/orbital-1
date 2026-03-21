"""Implementation of the Relu operator."""
import typing
import ibis
from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class ReluTranslator(Translator):
    def process(self) -> None:
        # https://onnx.ai/onnx/operators/onnx__Relu.html
        data = self._variables.consume(self.inputs[0])

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("Relu: The input must be a numeric column or a column group of numerics.")

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup({
                k: self._optimizer.fold_operation(ibis.greatest(ibis.literal(0.0), v))
                for k, v in data.items()
            })
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(ibis.greatest(ibis.literal(0.0), data))

        self.set_output(result)
