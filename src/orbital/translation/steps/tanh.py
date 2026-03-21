"""Implementation of the Tanh operator."""
import typing
import ibis
from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


def _tanh(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
    """Compute tanh(v) = (exp(2x) - 1) / (exp(2x) + 1).

    tanh() is not available as a built-in method on ibis NumericValue in ibis 12,
    so we use the equivalent exponential identity instead.
    """
    e2x = (v * ibis.literal(2.0)).exp()
    return (e2x - ibis.literal(1.0)) / (e2x + ibis.literal(1.0))


class TanhTranslator(Translator):
    def process(self) -> None:
        # https://onnx.ai/onnx/operators/onnx__Tanh.html
        data = self._variables.consume(self.inputs[0])

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("Tanh: The input must be a numeric column or a column group of numerics.")

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup({
                k: self._optimizer.fold_operation(_tanh(v))
                for k, v in data.items()
            })
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(_tanh(data))

        self.set_output(result)
