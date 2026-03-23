"""Implementation of the Softsign operator.

Softsign(x) = x / (1 + |x|)

References
----------
https://onnx.ai/onnx/operators/onnx__Softsign.html
"""

import ibis

from ._base_activation import UnaryActivationTranslator


class SoftsignTranslator(UnaryActivationTranslator):
    """Translate the ONNX Softsign operator: ``x / (1 + |x|)``."""

    # https://onnx.ai/onnx/operators/onnx__Softsign.html
    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        return v / (ibis.literal(1.0) + v.abs())
