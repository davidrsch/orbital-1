"""Implementation of the Exp operator.

Exp(x) = e^x

References
----------
https://onnx.ai/onnx/operators/onnx__Exp.html
"""

import ibis

from ._base_activation import UnaryActivationTranslator


class ExpTranslator(UnaryActivationTranslator):
    """Translate the ONNX Exp operator: ``e^x``."""

    # https://onnx.ai/onnx/operators/onnx__Exp.html
    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        return v.exp()
