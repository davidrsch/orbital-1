"""Implementation of the Abs operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class AbsTranslator(Translator):
    """Translate the ONNX Abs operator: absolute value element-wise."""

    def process(self) -> None:
        """Translate the Abs node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__Abs.html
        data = self._variables.consume(self.inputs[0])

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("Abs: The input must be a numeric column.")

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup(
                {k: self._optimizer.fold_operation(v.abs()) for k, v in data.items()}
            )
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(data.abs())

        self.set_output(result)
