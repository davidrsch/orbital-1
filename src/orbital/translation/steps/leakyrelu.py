"""Implementation of the LeakyRelu operator."""

import ibis

from ._base_activation import UnaryActivationTranslator


class LeakyReluTranslator(UnaryActivationTranslator):
    """Translate the ONNX LeakyRelu operator: ``x if x >= 0 else alpha*x``."""

    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        alpha = float(self._attributes.get("alpha", 0.01))
        return ibis.cases(
            (v >= ibis.literal(0.0), v),
            else_=ibis.literal(alpha) * v,
        )
