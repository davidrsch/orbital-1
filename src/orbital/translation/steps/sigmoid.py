"""Implementation of the Sigmoid operator."""

import ibis

from ._base_activation import UnaryActivationTranslator


class SigmoidTranslator(UnaryActivationTranslator):
    """Translate the ONNX Sigmoid operator: ``1 / (1 + exp(-x))``."""

    # https://onnx.ai/onnx/operators/onnx__Sigmoid.html
    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        return ibis.literal(1.0) / (ibis.literal(1.0) + (-v).exp())
