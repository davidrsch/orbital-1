"""Implementation of the Selu operator."""

import ibis

from ._base_activation import UnaryActivationTranslator

# ONNX-specified default constants for SELU
_SELU_ALPHA = 1.6732631921768192  # ONNX canonical default value
_SELU_GAMMA = 1.0507009873554805


class SeluTranslator(UnaryActivationTranslator):
    """Translate the ONNX Selu operator: scaled ELU activation."""

    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        alpha = float(self._attributes.get("alpha", _SELU_ALPHA))
        gamma = float(self._attributes.get("gamma", _SELU_GAMMA))
        return ibis.literal(gamma) * ibis.cases(
            (v > ibis.literal(0.0), v),
            else_=ibis.literal(alpha) * (v.exp() - ibis.literal(1.0)),
        )
