"""Implementation of the InstanceNormalization operator.

For a 1-D tabular input with C feature columns (no spatial dimensions),
InstanceNormalization normalises every *row* independently across all C columns:

    mean   = (1/C) * sum_c  x_c
    var    = (1/C) * sum_c  (x_c - mean)²
    y_c    = (x_c - mean) / sqrt(var + epsilon) * scale_c + bias_c

Because the per-row mean and variance are expressed as inline SQL expressions
the generated SQL can be verbose, but it is numerically correct.

References
----------
https://onnx.ai/onnx/operators/onnx__InstanceNormalization.html
"""

import ibis

from ._base_norm import NormTranslatorBase
from ..variables import NumericVariablesGroup, VariablesGroup


class InstanceNormalizationTranslator(NormTranslatorBase):
    """Translate the ONNX InstanceNormalization operator."""

    def process(self) -> None:
        """Translate the InstanceNormalization node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])
        scale, bias, epsilon = self._extract_scale_bias_epsilon(
            op_name="InstanceNormalization"
        )

        if bias is None:
            raise ValueError(
                "InstanceNormalization: bias must be a constant initializer."
            )

        if not isinstance(data, VariablesGroup):
            raise ValueError(
                "InstanceNormalization: input must be a column group representing "
                "the C channels (one SQL column per channel)."
            )

        data = NumericVariablesGroup(data)
        fields = list(data.keys())
        cols = list(data.values())
        n = len(cols)

        if len(scale) != n:
            raise ValueError(
                f"InstanceNormalization: scale length ({len(scale)}) "
                f"does not match number of channels ({n})."
            )
        if len(bias) != n:
            raise ValueError(
                f"InstanceNormalization: bias length ({len(bias)}) "
                f"does not match number of channels ({n})."
            )

        # Per-row mean across all C channels (inline expression).
        mean_expr = cols[0]
        for col in cols[1:]:
            mean_expr = mean_expr + col
        mean_expr = mean_expr / ibis.literal(float(n))

        # Per-row variance across all C channels (inline, duplicates mean_expr).
        var_terms = [(col - mean_expr) ** 2 for col in cols]
        var_expr = var_terms[0]
        for t in var_terms[1:]:
            var_expr = var_expr + t
        var_expr = var_expr / ibis.literal(float(n))

        std_expr = (var_expr + ibis.literal(epsilon)) ** ibis.literal(0.5)

        result = NumericVariablesGroup(
            {
                field: self._optimizer.fold_operation(
                    (cols[i] - mean_expr) / std_expr * ibis.literal(float(scale[i]))
                    + ibis.literal(float(bias[i]))
                )
                for i, field in enumerate(fields)
            }
        )
        self.set_output(result)
