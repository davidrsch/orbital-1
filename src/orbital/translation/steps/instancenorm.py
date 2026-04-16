"""Implementation of the InstanceNormalization operator.

For a 1-D tabular input with C feature columns (no spatial dimensions),
InstanceNormalization normalises every *row* independently across all C columns:

    mean   = (1/C) * sum_c  x_c
    var    = (1/C) * sum_c  (x_c - mean)²
    y_c    = (x_c - mean) / sqrt(var + epsilon) * scale_c + bias_c

In the ONNX spec, InstanceNorm reduces over the *spatial* dimensions for each
channel.  For a 2-D tabular input (batch × C features) there are no spatial
dimensions, so orbital treats the C feature columns as the instance and
normalises each row across all C columns.  This matches the behaviour of
MeanVarianceNormalization in the spatial-size-1 case but applies per-channel
affine transforms (scale/bias) from the original model.

Because the per-row mean and variance are expressed as inline SQL expressions
the generated SQL can be verbose, but it is numerically correct.

References
----------
https://onnx.ai/onnx/operators/onnx__InstanceNormalization.html
"""

import ibis

from ..variables import NumericVariablesGroup, VariablesGroup
from ._base_norm import NormTranslatorBase


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

        mean_expr, std_expr = self._compute_mean_std(cols, n, epsilon)

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
