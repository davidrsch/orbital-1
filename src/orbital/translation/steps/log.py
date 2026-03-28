"""Implementation of the Log operator.

Log(x) = ln(x)

References
----------
https://onnx.ai/onnx/operators/onnx__Log.html
"""

import ibis

from ._base_activation import UnaryActivationTranslator


class LogTranslator(UnaryActivationTranslator):
    """Translate the ONNX Log operator: ``ln(x)``."""

    # https://onnx.ai/onnx/operators/onnx__Log.html
    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        return v.ln()
