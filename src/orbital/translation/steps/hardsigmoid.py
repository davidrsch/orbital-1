"""Implementation of the HardSigmoid operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class HardSigmoidTranslator(Translator):
    """Translate the ONNX HardSigmoid operator: ``max(0, min(1, alpha*x + beta))``."""

    def process(self) -> None:
        """Translate the HardSigmoid node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__HardSigmoid.html
        # hard_sigmoid(x) = max(0, min(1, alpha*x + beta))
        data = self._variables.consume(self.inputs[0])
        alpha = float(self._attributes.get("alpha", 0.2))
        beta = float(self._attributes.get("beta", 0.5))

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("HardSigmoid: The input must be a numeric column.")

        def _hard_sigmoid(
            v: ibis.expr.types.NumericValue,
        ) -> ibis.expr.types.NumericValue:
            return ibis.greatest(
                ibis.literal(0.0),
                ibis.least(
                    ibis.literal(1.0),
                    ibis.literal(alpha) * v + ibis.literal(beta),
                ),
            )

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup(
                {
                    k: self._optimizer.fold_operation(_hard_sigmoid(v))
                    for k, v in data.items()
                }
            )
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(_hard_sigmoid(data))

        self.set_output(result)
