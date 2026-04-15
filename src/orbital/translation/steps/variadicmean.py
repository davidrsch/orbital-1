"""Implementation of the variadic ONNX Mean operator.

Computes the element-wise mean across N input tensors:

    y = (x_0 + x_1 + ... + x_{N-1}) / N

References
----------
https://onnx.ai/onnx/operators/onnx__Mean.html
"""

import ibis

from ._base_variadic_elementwise import _VariadicElementwiseTranslator


class VariadicMeanTranslator(_VariadicElementwiseTranslator):
    """Translate the ONNX variadic Mean operator."""

    def _combine(
        self,
        a: ibis.expr.types.NumericValue,
        b: ibis.expr.types.NumericValue,
    ) -> ibis.expr.types.NumericValue:
        return a + b

    def _post_process(
        self,
        expr: ibis.expr.types.NumericValue,
        n_inputs: int,
    ) -> ibis.expr.types.NumericValue:
        return expr / ibis.literal(float(n_inputs))
