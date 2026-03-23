"""Implementation of the GroupNormalization operator.

GroupNormalization divides the C input channels into *num_groups* groups and
normalises each group independently, analogous to LayerNorm within each group.

For a 1-D tabular input (C columns, no spatial dimensions):

    For each group g (channels g*C//G … (g+1)*C//G - 1):
        mean_g = (1/K) * sum_{c in g}  x_c          (K = C / num_groups)
        var_g  = (1/K) * sum_{c in g}  (x_c - mean_g)²
        y_c    = (x_c - mean_g) / sqrt(var_g + epsilon) * scale_c + bias_c

References
----------
https://onnx.ai/onnx/operators/onnx__GroupNormalization.html
"""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class GroupNormalizationTranslator(Translator):
    """Translate the ONNX GroupNormalization operator."""

    def process(self) -> None:
        """Translate the GroupNormalization node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])
        epsilon = float(self._attributes.get("epsilon", 1e-5))
        num_groups = int(self._attributes.get("num_groups", 1))

        scale = self._variables.get_initializer_value(self.inputs[1])
        bias = self._variables.get_initializer_value(self.inputs[2])

        if not isinstance(scale, (list, tuple)):
            raise ValueError(
                "GroupNormalization: scale must be a constant initializer."
            )
        if not isinstance(bias, (list, tuple)):
            raise ValueError("GroupNormalization: bias must be a constant initializer.")

        if not isinstance(data, VariablesGroup):
            raise ValueError(
                "GroupNormalization: input must be a column group representing "
                "the C channels (one SQL column per channel)."
            )

        data = NumericVariablesGroup(data)
        fields = list(data.keys())
        cols = list(data.values())
        n = len(cols)

        if n % num_groups != 0:
            raise ValueError(
                f"GroupNormalization: number of channels ({n}) must be divisible "
                f"by num_groups ({num_groups})."
            )
        if len(scale) != n:
            raise ValueError(
                f"GroupNormalization: scale length ({len(scale)}) "
                f"does not match number of channels ({n})."
            )
        if len(bias) != n:
            raise ValueError(
                f"GroupNormalization: bias length ({len(bias)}) "
                f"does not match number of channels ({n})."
            )

        group_size = n // num_groups
        result_exprs: dict[str, ibis.expr.types.NumericValue] = {}

        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            g_cols = cols[start:end]
            k = float(group_size)

            # Per-row mean for this group.
            mean_g = g_cols[0]
            for col in g_cols[1:]:
                mean_g = mean_g + col
            mean_g = mean_g / ibis.literal(k)

            # Per-row variance for this group.
            var_terms = [(col - mean_g) ** 2 for col in g_cols]
            var_g = var_terms[0]
            for t in var_terms[1:]:
                var_g = var_g + t
            var_g = var_g / ibis.literal(k)

            std_g = (var_g + ibis.literal(epsilon)) ** ibis.literal(0.5)

            for i_local, (field, col) in enumerate(zip(fields[start:end], g_cols)):
                c = start + i_local
                result_exprs[field] = self._optimizer.fold_operation(
                    (col - mean_g) / std_g * ibis.literal(float(scale[c]))
                    + ibis.literal(float(bias[c]))
                )

        self.set_output(NumericVariablesGroup(result_exprs))
