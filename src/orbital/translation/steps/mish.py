"""Implementation of the Mish operator.

Mish(x) = x * tanh(ln(1 + exp(x)))
        = x * tanh(softplus(x))

References
----------
https://onnx.ai/onnx/operators/onnx__Mish.html
"""

import ibis

from ._base_activation import UnaryActivationTranslator
from .tanh import _tanh


class MishTranslator(UnaryActivationTranslator):
    # https://onnx.ai/onnx/operators/onnx__Mish.html
    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        softplus = (ibis.literal(1.0) + v.exp()).ln()
        return v * _tanh(softplus)
