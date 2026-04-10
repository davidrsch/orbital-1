"""Implementation of the SkipLayerNormalization contrib operator (com.microsoft).

``SkipLayerNormalization`` is equivalent to ``LayerNorm(input + skip)``:

    intermediate_i = input_i + skip_i
    mean           = (1/H) * sum_i  intermediate_i
    var            = (1/H) * sum_i  (intermediate_i - mean)²
    output_i       = (intermediate_i - mean) / sqrt(var + epsilon) * gamma_i + beta_i

Operator inputs
---------------
0  input  : runtime tensor of shape [batch, seq_len, hidden_size]
1  skip   : runtime tensor, same shape as input
2  gamma  : constant initializer of shape [hidden_size]
3  beta   : constant initializer of shape [hidden_size] (optional)

Attributes
----------
epsilon : float, default 1e-12

References
----------
https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md
"""

import ibis

from ._base_norm import NormTranslatorBase
from ..variables import NumericVariablesGroup, VariablesGroup


class SkipLayerNormalizationTranslator(NormTranslatorBase):
    """Translate the ``com.microsoft.SkipLayerNormalization`` contrib operator."""

    def process(self) -> None:
        """Translate the SkipLayerNormalization node, writing result to the graph."""
        input_data = self._variables.consume(self.inputs[0])
        skip_data = self._variables.consume(self.inputs[1])
        epsilon = float(self._attributes.get("epsilon", 1e-12))

        gamma = self._variables.get_initializer_value(self.inputs[2])
        if not isinstance(gamma, (list, tuple)):
            raise ValueError(
                "SkipLayerNormalization: gamma (inputs[2]) must be a constant initializer."
            )

        beta: list | None = None
        if len(self.inputs) > 3 and self.inputs[3]:
            beta_raw = self._variables.get_initializer_value(self.inputs[3])
            if beta_raw is not None:
                if not isinstance(beta_raw, (list, tuple)):
                    raise ValueError(
                        "SkipLayerNormalization: beta (inputs[3]) must be a constant initializer."
                    )
                beta = list(beta_raw)

        if not isinstance(input_data, VariablesGroup):
            raise ValueError(
                "SkipLayerNormalization: input (inputs[0]) must be a column group "
                "representing the hidden dimensions (one SQL column per dimension)."
            )
        if not isinstance(skip_data, VariablesGroup):
            raise ValueError(
                "SkipLayerNormalization: skip (inputs[1]) must be a column group "
                "representing the hidden dimensions (one SQL column per dimension)."
            )

        input_data = NumericVariablesGroup(input_data)
        skip_data = NumericVariablesGroup(skip_data)
        fields = list(input_data.keys())
        in_cols = list(input_data.values())
        skip_cols = list(skip_data.values())
        n = len(in_cols)

        if len(skip_cols) != n:
            raise ValueError(
                f"SkipLayerNormalization: input and skip must have the same number "
                f"of columns; got {n} vs {len(skip_cols)}."
            )
        if len(gamma) != n:
            raise ValueError(
                f"SkipLayerNormalization: gamma length ({len(gamma)}) "
                f"does not match number of hidden dimensions ({n})."
            )
        if beta is not None and len(beta) != n:
            raise ValueError(
                f"SkipLayerNormalization: beta length ({len(beta)}) "
                f"does not match number of hidden dimensions ({n})."
            )

        # Step 1: input + skip
        inter_cols = [in_cols[i] + skip_cols[i] for i in range(n)]

        # Step 2: per-row mean and std using the shared base helper
        mean_expr, std_expr = self._compute_mean_std(inter_cols, n, epsilon)

        # Step 3: normalise with gamma and beta
        result = NumericVariablesGroup(
            {
                field: self._optimizer.fold_operation(
                    (inter_cols[i] - mean_expr) / std_expr * ibis.literal(float(gamma[i]))
                    + ibis.literal(float(beta[i]) if beta is not None else 0.0)
                )
                for i, field in enumerate(fields)
            }
        )
        self.set_output(result)
