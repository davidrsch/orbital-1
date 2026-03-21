"""Implementation of the Sigmoid operator."""
import typing
import ibis
from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class SigmoidTranslator(Translator):
    def process(self) -> None:
        # https://onnx.ai/onnx/operators/onnx__Sigmoid.html
        data = self._variables.consume(self.inputs[0])

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("Sigmoid: The input must be a numeric column or a column group of numerics.")

        def _sigmoid(v):
            return ibis.literal(1.0) / (ibis.literal(1.0) + (-v).exp())

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup({
                k: self._optimizer.fold_operation(_sigmoid(v))
                for k, v in data.items()
            })
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(_sigmoid(data))

        self.set_output(result)
