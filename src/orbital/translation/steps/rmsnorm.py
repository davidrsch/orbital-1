"""Implementation of the RMSNormalization operator (ONNX opset 23+).

For a tabular row with C feature columns (axis=-1), RMSNormalization
normalises each row using only the root-mean-square, without mean-centring:

    rms   = sqrt((1/C) * sum_c  x_c^2  +  epsilon)
    y_c   = (x_c / rms) * scale_c

Unlike LayerNormalization there is no mean subtraction and no bias term.

References
----------
https://onnx.ai/onnx/operators/onnx__RMSNormalization.html
"""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class RMSNormalizationTranslator(Translator):
    """Translate the ONNX RMSNormalization operator (opset 23+)."""

    def process(self) -> None:
        """Translate the RMSNormalization node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])
        epsilon = float(self._attributes.get("epsilon", 1e-5))

        scale = self._variables.get_initializer_value(self.inputs[1])

        if not isinstance(scale, (list, tuple)):
            raise ValueError(
                "RMSNormalization: scale (inputs[1]) must be a constant initializer."
            )

        if not isinstance(data, VariablesGroup):
            raise ValueError(
                "RMSNormalization: input must be a column group representing "
                "the C features (one SQL column per feature)."
            )

        data = NumericVariablesGroup(data)
        fields = list(data.keys())
        cols = list(data.values())
        n = len(cols)

        axis = int(self._attributes.get("axis", -1))
        if axis not in (-1, n - 1):
            raise NotImplementedError(
                f"RMSNormalization: only last-axis normalisation (axis=-1) is "
                f"supported; got axis={axis} with {n} features."
            )

        if len(scale) != n:
            raise ValueError(
                f"RMSNormalization: scale length ({len(scale)}) "
                f"does not match number of features ({n})."
            )

        # Per-row mean of squares: (1/C) * sum_c(x_c^2)
        sq_sum = cols[0] ** 2
        for col in cols[1:]:
            sq_sum = sq_sum + col**2
        mean_sq = sq_sum / ibis.literal(float(n))

        # rms = sqrt(mean_sq + epsilon)
        rms_expr = (mean_sq + ibis.literal(epsilon)) ** ibis.literal(0.5)

        result = NumericVariablesGroup(
            {
                field: self._optimizer.fold_operation(
                    col / rms_expr * ibis.literal(float(scale[i]))
                )
                for i, (field, col) in enumerate(zip(fields, cols))
            }
        )

        self.set_output(result)
