"""Implementation of the Tanh operator."""

import ibis

from ._base_activation import UnaryActivationTranslator


def _tanh(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
    """Compute tanh(v) using the exponential identity with large-input clamping.

    tanh() is not available as a built-in method on ibis NumericValue in ibis 12,
    so we use the equivalent exponential identity.  For |v| > 20 the identity
    overflows to Inf/NaN on some SQL engines, so we clamp to ±1.0 first.
    """
    e2x = (v * ibis.literal(2.0)).exp()
    raw = (e2x - ibis.literal(1.0)) / (e2x + ibis.literal(1.0))
    return ibis.ifelse(
        v > ibis.literal(20.0),
        ibis.literal(1.0),
        ibis.ifelse(v < ibis.literal(-20.0), ibis.literal(-1.0), raw),
    )


class TanhTranslator(UnaryActivationTranslator):
    """Translate the ONNX Tanh operator."""

    # https://onnx.ai/onnx/operators/onnx__Tanh.html
    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        return _tanh(v)
