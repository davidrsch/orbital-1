"""Implementation of the variadic ONNX Min operator.

Computes the element-wise minimum across N input tensors:

    y = min(x_0, x_1, ..., x_{N-1})

Note: this is the *variadic* Min (N tensors → 1 tensor), not ReduceMin
which reduces a single tensor along an axis.

References
----------
https://onnx.ai/onnx/operators/onnx__Min.html
"""

import ibis

from ._base_variadic_elementwise import _VariadicElementwiseTranslator


class VariadicMinTranslator(_VariadicElementwiseTranslator):
    """Translate the ONNX variadic Min operator."""

    def _combine(
        self,
        a: ibis.expr.types.NumericValue,
        b: ibis.expr.types.NumericValue,
    ) -> ibis.expr.types.NumericValue:
        return ibis.least(a, b)
