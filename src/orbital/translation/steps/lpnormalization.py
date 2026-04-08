"""Implementation of the LpNormalization operator.

For a tabular row with C feature columns, LpNormalization normalises the row
using the Lp norm along the last axis:

    p=1:  y_c = x_c / sum_c |x_c|          (L1 normalisation)
    p=2:  y_c = x_c / sqrt(sum_c x_c^2)    (L2 normalisation, default)

Only p=1 and p=2 are supported.  Only axis=-1 / axis=1 (the feature axis)
is supported for the tabular case.

References
----------
https://onnx.ai/onnx/operators/onnx__LpNormalization.html
"""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class LpNormalizationTranslator(Translator):
    """Translate the ONNX LpNormalization operator."""

    def process(self) -> None:
        """Translate the LpNormalization node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])

        p = int(self._attributes.get("p", 2))
        axis = int(self._attributes.get("axis", -1))

        if p not in (1, 2):
            raise NotImplementedError(
                f"LpNormalization: only p=1 and p=2 are supported; got p={p}."
            )

        if not isinstance(data, VariablesGroup):
            raise ValueError(
                "LpNormalization: input must be a column group representing "
                "the C features (one SQL column per feature)."
            )

        data = NumericVariablesGroup(data)
        fields = list(data.keys())
        cols = list(data.values())
        n = len(cols)

        if axis not in (-1, n - 1):
            raise NotImplementedError(
                f"LpNormalization: only last-axis normalisation (axis=-1) is "
                f"supported; got axis={axis} with {n} features."
            )

        if p == 1:
            # norm = sum_c |x_c|
            norm_expr = cols[0].abs()
            for col in cols[1:]:
                norm_expr = norm_expr + col.abs()
        else:
            # p == 2: norm = sqrt(sum_c x_c^2)
            norm_expr = cols[0] ** 2
            for col in cols[1:]:
                norm_expr = norm_expr + col ** 2
            norm_expr = norm_expr ** ibis.literal(0.5)

        result = NumericVariablesGroup(
            {
                field: self._optimizer.fold_operation(
                    col / norm_expr
                )
                for field, col in zip(fields, cols)
            }
        )

        self.set_output(result)
