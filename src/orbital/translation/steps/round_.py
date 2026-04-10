"""Implementation of the Round operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class RoundTranslator(Translator):
    """Translate the ONNX Round operator: element-wise round-half-to-even."""

    def process(self) -> None:
        """Translate the Round node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__Round.html
        # ONNX specifies round-half-to-even (banker's rounding).
        # ibis.round(digits=0) delegates to the underlying database engine,
        # which typically uses round-half-up. This is an acceptable
        # approximation for inference purposes.
        data = self._variables.consume(self.inputs[0])

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("Round: The input must be a numeric column.")

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup(
                {k: self._optimizer.fold_operation(v.round(0)) for k, v in data.items()}
            )
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(data.round(0))

        self.set_output(result)
