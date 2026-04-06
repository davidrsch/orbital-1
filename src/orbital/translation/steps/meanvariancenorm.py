"""Implementation of the MeanVarianceNormalization operator (ONNX opset 9).

Normalises the input tensor element-wise using per-row mean and variance:

    mean  = (1/C) * sum_c x_c
    var   = (1/C) * sum_c (x_c - mean)^2
    y_c   = (x_c - mean) / sqrt(var + epsilon)

The ``epsilon`` attribute is read from the node; the ONNX spec default is
``1e-5``.  Only normalisation over the feature axis (all columns in a
VariablesGroup) is supported; the ONNX ``axes`` attribute is accepted only when
it reduces to feature-axis normalisation.

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
        epsilon: float = float(self._attributes.get("epsilon", 1e-5))
        data = self._variables.consume(self.inputs[0])

        if not isinstance(data, VariablesGroup):
            raise ValueError(
                "MeanVarianceNormalization: input must be a column group."
            )

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

        std_expr = (var_expr + ibis.literal(epsilon)) ** ibis.literal(0.5)

        result = NumericVariablesGroup(
            {
                field: self._optimizer.fold_operation((cols[i] - mean_expr) / std_expr)
                for i, field in enumerate(fields)
            }
        )
        self.set_output(result)
