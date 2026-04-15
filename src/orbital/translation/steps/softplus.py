"""Implementation of the Softplus operator.

Softplus(x) = ln(1 + exp(x))

For x > 20 the naive form overflows; we use the numerically stable
identity: softplus(x) = x + ln(1 + exp(-x)) ≈ x for large x.

References
----------
https://onnx.ai/onnx/operators/onnx__Softplus.html
"""

import ibis

from ._base_activation import UnaryActivationTranslator


class SoftplusTranslator(UnaryActivationTranslator):
    """Translate the ONNX Softplus operator: ``ln(1 + exp(x))``."""

    # https://onnx.ai/onnx/operators/onnx__Softplus.html
    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        # For x > 20, exp(x) overflows; softplus(x) ≈ x for large x.
        stable = (ibis.literal(1.0) + v.exp()).ln()
        return ibis.ifelse(v > ibis.literal(20.0), v, stable)
