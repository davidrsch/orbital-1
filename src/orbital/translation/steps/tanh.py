"""Implementation of the Tanh operator."""

import ibis

from ._base_activation import UnaryActivationTranslator


def _tanh(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
    """Compute tanh(v) = (exp(2x) - 1) / (exp(2x) + 1).

    tanh() is not available as a built-in method on ibis NumericValue in ibis 12,
    so we use the equivalent exponential identity instead.
    """
    e2x = (v * ibis.literal(2.0)).exp()
    return (e2x - ibis.literal(1.0)) / (e2x + ibis.literal(1.0))


class TanhTranslator(UnaryActivationTranslator):
    """Translate the ONNX Tanh operator via polynomial approximation."""

    # https://onnx.ai/onnx/operators/onnx__Tanh.html
    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        return _tanh(v)
