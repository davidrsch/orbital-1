"""Implementation of the LayerNormalization operator (ONNX opset 17+).

For a tabular row with C feature columns (axis=-1), LayerNormalization
normalises each row independently:

    mean   = (1/C) * sum_c  x_c
    var    = (1/C) * sum_c  (x_c - mean)²
    y_c    = (x_c - mean) / sqrt(var + epsilon) * scale_c + bias_c

The implementation follows the same symbolic per-row formula used in
:class:`~orbital.translation.steps.instancenorm.InstanceNormalizationTranslator`.

References
----------
https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
"""

import ibis

from ._base_norm import NormTranslatorBase
from ..variables import NumericVariablesGroup, VariablesGroup


class LayerNormalizationTranslator(NormTranslatorBase):
    """Translate the ONNX LayerNormalization operator (opset 17+)."""

    def process(self) -> None:
        """Translate the LayerNormalization node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])
        # bias_idx=2 with optional presence — base class guards via len(self.inputs)
        scale, bias, epsilon = self._extract_scale_bias_epsilon(
            op_name="LayerNormalization"
        )

        if not isinstance(data, VariablesGroup):
            raise ValueError(
                "LayerNormalization: input must be a column group representing "
                "the C features (one SQL column per feature)."
            )

        data = NumericVariablesGroup(data)
        fields = list(data.keys())
        cols = list(data.values())
        n = len(cols)

        axis = int(self._attributes.get("axis", -1))
        if axis not in (-1, n - 1):
            raise NotImplementedError(
                f"LayerNormalization: only last-axis normalisation (axis=-1) is "
                f"supported; got axis={axis} with {n} features."
            )

        if len(scale) != n:
            raise ValueError(
                f"LayerNormalization: scale length ({len(scale)}) "
                f"does not match number of features ({n})."
            )
        if bias is not None and len(bias) != n:
            raise ValueError(
                f"LayerNormalization: bias length ({len(bias)}) "
                f"does not match number of features ({n})."
            )

        # Per-row mean across all features (inline SQL expression)
        mean_expr = cols[0]
        for col in cols[1:]:
            mean_expr = mean_expr + col
        mean_expr = mean_expr / ibis.literal(float(n))

        # Per-row variance (inline SQL expression)
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
                    + ibis.literal(float(bias[i]) if bias is not None else 0.0)
                )
                for i, field in enumerate(fields)
            }
        )
        self.set_output(result)
