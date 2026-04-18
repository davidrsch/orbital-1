"""Implementation of the Elu operator."""

import ibis

from .._activation_defaults import ELU_ALPHA
from ._base_activation import UnaryActivationTranslator


class EluTranslator(UnaryActivationTranslator):
    """Translate the ONNX Elu operator: ``x if x >= 0 else alpha*(exp(x)-1)``."""

    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        alpha = float(self._attributes.get("alpha", ELU_ALPHA))
        return ibis.cases(
            (v >= ibis.literal(0.0), v),
            else_=ibis.literal(alpha) * (v.exp() - ibis.literal(1.0)),
        )
