"""Implementation of the Swish operator (ONNX opset 24).

Swish(x) = x * sigmoid(x) = x / (1 + e^{-x})

This is the native ONNX Swish operator introduced in opset 24.
It is equivalent to SiLU (Sigmoid Linear Unit) from PyTorch and
the ``silu``/``swish`` activation in Keras.

References
----------
https://onnx.ai/onnx/operators/onnx__Swish.html
"""

import ibis

from ._base_activation import UnaryActivationTranslator


class SwishTranslator(UnaryActivationTranslator):
    """Translate the ONNX Swish operator: ``x * sigmoid(x)``."""

    # https://onnx.ai/onnx/operators/onnx__Swish.html
    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        sigmoid = ibis.literal(1.0) / (ibis.literal(1.0) + (-v).exp())
        return v * sigmoid
