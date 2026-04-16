"""Implementation of the MeanVarianceNormalization operator (ONNX opset 9).

Normalises the input tensor element-wise using per-row mean and variance:

    mean  = (1/C) * sum_c x_c
    var   = (1/C) * sum_c (x_c - mean)^2
    y_c   = (x_c - mean) / sqrt(var)

Only normalisation over the feature axis (all columns in a VariablesGroup) is
supported; the ONNX ``axes`` attribute must be absent (default all-axes
normalisation).

References
----------
https://onnx.ai/onnx/operators/onnx__MeanVarianceNormalization.html
"""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class MeanVarianceNormalizationTranslator(Translator):
    """Translate the ONNX MeanVarianceNormalization operator."""

    def process(self) -> None:
        """Translate the MeanVarianceNormalization node."""
        axes = self._attributes.get("axes", None)
        if axes is not None and list(axes) != [1]:
            # ONNX default is axes=[0, 2, 3] for NCHW; only last-axis normalisation is supported
            # For 2D tabular input (N, C), axes=[1] means "normalise along the feature axis",
            # which is semantically identical to the default behaviour.
            # Raise for any other explicit axes specification.
            raise NotImplementedError(
                f"MeanVarianceNormalization: axes={axes} is not supported; "
                "only default (all axes) normalisation is supported."
            )
        data = self._variables.consume(self.inputs[0])

        if not isinstance(data, VariablesGroup):
            raise ValueError("MeanVarianceNormalization: input must be a column group.")

        data = NumericVariablesGroup(data)
        fields = list(data.keys())
        cols = list(data.values())
        n = len(cols)

        mean_expr = cols[0]
        for col in cols[1:]:
            mean_expr = mean_expr + col
        mean_expr = mean_expr / ibis.literal(float(n))

        var_terms = [(col - mean_expr) ** 2 for col in cols]
        var_expr = var_terms[0]
        for t in var_terms[1:]:
            var_expr = var_expr + t
        var_expr = var_expr / ibis.literal(float(n))

        # ONNX spec: MeanVarianceNormalization does not have an epsilon attribute.
        # The spec uses a fixed epsilon of 1e-9 inside the square root.
        epsilon = 1e-9
        denom_expr = (var_expr + ibis.literal(epsilon)) ** ibis.literal(0.5)

        result = NumericVariablesGroup(
            {
                field: self._optimizer.fold_operation(
                    (cols[i] - mean_expr) / denom_expr
                )
                for i, field in enumerate(fields)
            }
        )
        self.set_output(result)
