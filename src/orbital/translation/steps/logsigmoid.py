"""Implementation of the LogSigmoid operator (used by PyTorch MLP exports).

LogSigmoid(x) = ln(sigmoid(x)) = ln(1 / (1 + exp(-x))) = -ln(1 + exp(-x))

Note: LogSigmoid is not a standard ONNX op by that name, but PyTorch exports it
as a compound expression.  Some frameworks also export it as a standalone custom
op named "LogSigmoid".  This translator handles that case.

References
----------
https://pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html
"""

import ibis

from ._base_activation import UnaryActivationTranslator


class LogSigmoidTranslator(UnaryActivationTranslator):
    """Translate the ONNX LogSigmoid operator: ``log(sigmoid(x))``."""

    def _apply(self, v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
        return -(ibis.literal(1.0) + (-v).exp()).ln()
