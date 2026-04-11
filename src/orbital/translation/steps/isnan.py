"""Implementation of the IsNaN operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class IsNaNTranslator(Translator):
    """Translate the ONNX IsNaN operator: element-wise boolean NaN check."""

    def process(self) -> None:
        """Translate the IsNaN node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__IsNaN.html
        # ibis .isnan() returns True only for IEEE NaN values (not NULL).
        data = self._variables.consume(self.inputs[0])

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("IsNaN: The input must be a numeric column.")

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup(
                {k: self._optimizer.fold_operation(v.isnan()) for k, v in data.items()}
            )
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(data.isnan())

        self.set_output(result)
