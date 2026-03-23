"""Implementation of the Softplus operator.

Softplus(x) = ln(1 + exp(x))

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
        return (ibis.literal(1.0) + v.exp()).ln()
