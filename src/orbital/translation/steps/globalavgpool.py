"""Implementation of the GlobalAveragePool operator.

In the SQL/tabular context, each input row represents one sample.  A
``VariablesGroup`` of C columns corresponds to C channels (or sequence
positions), and GlobalAveragePool reduces them to their per-row mean —
collapsing the spatial/sequence dimension to a single scalar.

For a single-column input (no group), the operation is the identity.

References
----------
https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html
"""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class GlobalAveragePoolTranslator(Translator):
    """Translate the ONNX GlobalAveragePool operator: per-row mean across all channels."""

    def process(self) -> None:
        """Translate the GlobalAveragePool node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            cols = list(data.values())
            n = len(cols)
            if n == 0:
                raise ValueError(
                    "GlobalAveragePool: input group must have at least one column."
                )
            mean_expr: ibis.expr.types.NumericValue = cols[0]
            for col in cols[1:]:
                mean_expr = mean_expr + col
            mean_expr = mean_expr / ibis.literal(float(n))
            self.set_output(self._optimizer.fold_operation(mean_expr))
        else:
            # Single column: already reduced.
            self.set_output(data)
