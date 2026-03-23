"""Implementation of the ThresholdedRelu operator.

ThresholdedRelu(x) = x  if x > alpha, else 0
Default alpha = 1.0

References
----------
https://onnx.ai/onnx/operators/onnx__ThresholdedRelu.html
"""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class ThresholdedReluTranslator(Translator):
    """Translate the ONNX ThresholdedRelu operator: ``x if x > alpha else 0``."""

    def process(self) -> None:
        """Translate the ThresholdedRelu node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__ThresholdedRelu.html
        data = self._variables.consume(self.inputs[0])
        alpha = float(self._attributes.get("alpha", 1.0))

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("ThresholdedRelu: The input must be a numeric column.")

        def _threlu(
            v: ibis.expr.types.NumericValue,
        ) -> ibis.expr.types.NumericValue:
            return ibis.cases(
                (v > ibis.literal(alpha), v),
                else_=ibis.literal(0.0),
            )

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup(
                {k: self._optimizer.fold_operation(_threlu(v)) for k, v in data.items()}
            )
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(_threlu(data))

        self.set_output(result)
