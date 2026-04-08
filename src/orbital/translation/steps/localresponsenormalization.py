"""Implementation of the LRN (Local Response Normalization) operator.

For a tabular row with C feature columns, LRN normalises each element using
a sliding window of neighbouring channels:

    y_c = x_c / (k + (alpha / size) * sum_{j in window(c)} x_j^2) ^ beta

where the window is:
    max(0, c - floor((size-1)/2))  …  min(C-1, c + ceil((size-1)/2))

Default parameter values follow the ONNX specification:
    alpha = 0.0001,  beta = 0.75,  bias (k) = 1.0,  size = required

Only the cross-channel axis (last axis) is supported for the tabular case.

References
----------
https://onnx.ai/onnx/operators/onnx__LRN.html
"""

import math

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class LRNTranslator(Translator):
    """Translate the ONNX LRN (Local Response Normalization) operator."""

    def process(self) -> None:
        """Translate the LRN node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])

        size = int(self._attributes["size"])
        alpha = float(self._attributes.get("alpha", 0.0001))
        beta = float(self._attributes.get("beta", 0.75))
        k = float(self._attributes.get("bias", 1.0))
        alpha_over_size = alpha / size

        if not isinstance(data, VariablesGroup):
            raise ValueError(
                "LRN: input must be a column group representing "
                "the C features (one SQL column per feature)."
            )

        data = NumericVariablesGroup(data)
        fields = list(data.keys())
        cols = list(data.values())
        n = len(cols)

        half_left = (size - 1) // 2
        half_right = math.ceil((size - 1) / 2)

        result_exprs: dict[str, ibis.expr.types.NumericValue] = {}
        for c, (field, col) in enumerate(zip(fields, cols)):
            lo = max(0, c - half_left)
            hi = min(n - 1, c + half_right)
            window_cols = cols[lo : hi + 1]

            sq_sum = window_cols[0] ** 2
            for wc in window_cols[1:]:
                sq_sum = sq_sum + wc ** 2

            denom = (
                ibis.literal(k) + ibis.literal(alpha_over_size) * sq_sum
            ) ** ibis.literal(beta)

            result_exprs[field] = self._optimizer.fold_operation(col / denom)

        self.set_output(NumericVariablesGroup(result_exprs))
