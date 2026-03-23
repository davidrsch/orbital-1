"""Implementation of the HardTanh (Hardtanh) operator.

HardTanh(x) = -1   if x < -1
            =  x   if -1 <= x <= 1
            =  1   if x > 1

Note: This is also equivalent to Clip(x, -1, 1), but exported by some
frameworks as a named activation node.

References
----------
https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html
"""

import ibis

from ._base_activation import UnaryActivationTranslator


class HardTanhTranslator(UnaryActivationTranslator):
    """Translate the ONNX HardTanh operator: ``clip(x, -1, 1)``."""

    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        return ibis.cases(
            (v < ibis.literal(-1.0), ibis.literal(-1.0)),
            (v > ibis.literal(1.0), ibis.literal(1.0)),
            else_=v,
        )
