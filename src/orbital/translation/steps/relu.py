"""Implementation of the Relu operator."""

import ibis

from ._base_activation import UnaryActivationTranslator


class ReluTranslator(UnaryActivationTranslator):
    """Translate the ONNX Relu operator: ``max(0, x)``."""

    # https://onnx.ai/onnx/operators/onnx__Relu.html
    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        return ibis.greatest(ibis.literal(0.0), v)
