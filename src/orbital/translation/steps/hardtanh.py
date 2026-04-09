"""Implementation of the HardTanh (Hardtanh) operator.

HardTanh(x) = min_val   if x < min_val
            =  x        if min_val <= x <= max_val
            = max_val   if x > max_val

The default bounds are min_val=-1 and max_val=1, but PyTorch nn.Hardtanh
allows arbitrary bounds that are exported as ONNX node attributes ``min``
and ``max``.

References
----------
https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html
"""

import ibis

from ._base_activation import UnaryActivationTranslator


class HardTanhTranslator(UnaryActivationTranslator):
    """Translate the ONNX HardTanh operator: ``clip(x, min_val, max_val)``."""

    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        min_val = float(self._attributes.get("min", -1.0))
        max_val = float(self._attributes.get("max", 1.0))
        return ibis.cases(
            (v < ibis.literal(min_val), ibis.literal(min_val)),
            (v > ibis.literal(max_val), ibis.literal(max_val)),
            else_=v,
        )
