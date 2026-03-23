"""Implementation of the Neg operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class NegTranslator(Translator):
    """Translate the ONNX Neg operator: element-wise negation."""

    def process(self) -> None:
        """Translate the Neg node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__Neg.html
        data = self._variables.consume(self.inputs[0])

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("Neg: The input must be a numeric column.")

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup(
                {k: self._optimizer.fold_operation(-v) for k, v in data.items()}
            )
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(-data)

        self.set_output(result)
