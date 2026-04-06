"""Implementation of the Selu operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup

# ONNX-specified default constants for SELU
_SELU_ALPHA = 1.6732631921768192  # ONNX spec canonical value
_SELU_GAMMA = 1.0507009873554805


class SeluTranslator(Translator):
    """Translate the ONNX Selu operator: scaled ELU activation."""

    def process(self) -> None:
        """Translate the Selu node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__Selu.html
        # selu(x) = gamma * (x if x > 0 else alpha * (exp(x) - 1))
        data = self._variables.consume(self.inputs[0])
        alpha = float(self._attributes.get("alpha", _SELU_ALPHA))
        gamma = float(self._attributes.get("gamma", _SELU_GAMMA))

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("Selu: The input must be a numeric column.")

        def _selu(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
            return ibis.literal(gamma) * ibis.cases(
                (v > ibis.literal(0.0), v),
                else_=ibis.literal(alpha) * (v.exp() - ibis.literal(1.0)),
            )

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup(
                {k: self._optimizer.fold_operation(_selu(v)) for k, v in data.items()}
            )
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(_selu(data))

        self.set_output(result)
