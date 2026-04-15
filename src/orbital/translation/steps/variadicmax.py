"""Implementation of the variadic ONNX Max operator.

Computes the element-wise maximum across N input tensors:

    y = max(x_0, x_1, ..., x_{N-1})

Note: this is the *variadic* Max (N tensors → 1 tensor), not ReduceMax
which reduces a single tensor along an axis.

References
----------
https://onnx.ai/onnx/operators/onnx__Max.html
"""

import ibis

from ._base_variadic_elementwise import _VariadicElementwiseTranslator


class VariadicMaxTranslator(_VariadicElementwiseTranslator):
    """Translate the ONNX variadic Max operator."""

    def _combine(
        self,
        a: ibis.expr.types.NumericValue,
        b: ibis.expr.types.NumericValue,
    ) -> ibis.expr.types.NumericValue:
        return ibis.greatest(a, b)
