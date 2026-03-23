"""Implementation of the AveragePool operator.

For the SQL/tabular context, AveragePool is supported when:

- ``kernel_shape`` has all dimensions equal to 1 (pass-through), or
- The input is a ``VariablesGroup`` of *L* columns and ``kernel_shape = [L]``
  (global 1-D average pool, returning the per-row mean across all L columns).

Other shapes raise ``NotImplementedError``.

References
----------
https://onnx.ai/onnx/operators/onnx__AveragePool.html
"""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class AveragePoolTranslator(Translator):
    """Translate the ONNX AveragePool operator for the SQL/tabular context."""

    def process(self) -> None:
        """Translate the AveragePool node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])
        kernel_shape: list[int] = list(self._attributes.get("kernel_shape", [1]))

        # Trivial case: kernel = 1 everywhere → identity.
        if all(k == 1 for k in kernel_shape):
            self.set_output(data)
            return

        if not isinstance(data, VariablesGroup):
            raise NotImplementedError(
                f"AveragePool: kernel_shape={kernel_shape} is not supported for a "
                "single-column input."
            )

        data = NumericVariablesGroup(data)
        cols = list(data.values())
        n = len(cols)

        # Global average pool: kernel covers all columns.
        if len(kernel_shape) == 1 and kernel_shape[0] == n:
            mean_expr: ibis.expr.types.NumericValue = cols[0]
            for col in cols[1:]:
                mean_expr = mean_expr + col
            mean_expr = mean_expr / ibis.literal(float(n))
            self.set_output(self._optimizer.fold_operation(mean_expr))
            return

        raise NotImplementedError(
            f"AveragePool: kernel_shape={kernel_shape} with {n} input column(s) is "
            "not supported in the SQL context.  Only kernel_shape=[1] (identity) "
            "or kernel_shape=[n_cols] (global average) are supported."
        )
