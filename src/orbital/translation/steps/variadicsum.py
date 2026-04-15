"""Implementation of the variadic ONNX Sum operator.

Computes the element-wise sum of N input tensors:

    y = x_0 + x_1 + ... + x_{N-1}

References
----------
https://onnx.ai/onnx/operators/onnx__Sum.html
"""

import ibis

from ._base_variadic_elementwise import _VariadicElementwiseTranslator


class VariadicSumTranslator(_VariadicElementwiseTranslator):
    """Translate the ONNX variadic Sum operator."""

    def _combine(
        self,
        a: ibis.expr.types.NumericValue,
        b: ibis.expr.types.NumericValue,
    ) -> ibis.expr.types.NumericValue:
        return a + b
