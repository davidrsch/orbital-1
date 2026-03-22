"""Implementation of the Sigmoid operator."""
import ibis
from ._base_activation import UnaryActivationTranslator


class SigmoidTranslator(UnaryActivationTranslator):
    # https://onnx.ai/onnx/operators/onnx__Sigmoid.html
    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        return ibis.literal(1.0) / (ibis.literal(1.0) + (-v).exp())
