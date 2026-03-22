"""Implementation of the Relu operator."""
import ibis
from ._base_activation import UnaryActivationTranslator


class ReluTranslator(UnaryActivationTranslator):
    # https://onnx.ai/onnx/operators/onnx__Relu.html
    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        return ibis.greatest(ibis.literal(0.0), v)
