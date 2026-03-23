"""Implementation of the LeakyRelu operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class LeakyReluTranslator(Translator):
    """Translate the ONNX LeakyRelu operator: ``x if x >= 0 else alpha*x``."""

    def process(self) -> None:
        """Translate the LeakyRelu node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__LeakyRelu.html
        # leaky_relu(x) = x if x >= 0 else alpha * x
        data = self._variables.consume(self.inputs[0])
        alpha = float(self._attributes.get("alpha", 0.01))

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("LeakyRelu: The input must be a numeric column.")

        def _leaky_relu(
            v: ibis.expr.types.NumericValue,
        ) -> ibis.expr.types.NumericValue:
            return ibis.cases(
                (v >= ibis.literal(0.0), v),
                else_=ibis.literal(alpha) * v,
            )

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup(
                {
                    k: self._optimizer.fold_operation(_leaky_relu(v))
                    for k, v in data.items()
                }
            )
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(_leaky_relu(data))

        self.set_output(result)
