"""Implementation of the MaxPool operator.

For the SQL/tabular context, MaxPool is supported when:

- ``kernel_shape`` has all dimensions equal to 1 (pass-through), or
- The input is a ``VariablesGroup`` of *L* columns and ``kernel_shape = [L]``
  (global 1-D max pool, returning the per-row max across all L columns).

Other shapes raise ``NotImplementedError``.

References
----------
https://onnx.ai/onnx/operators/onnx__MaxPool.html
"""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class MaxPoolTranslator(Translator):
    def process(self) -> None:
        data = self._variables.consume(self.inputs[0])
        kernel_shape: list[int] = list(
            self._attributes.get("kernel_shape", [1])
        )

        # Trivial case: kernel = 1 everywhere → identity.
        if all(k == 1 for k in kernel_shape):
            self.set_output(data)
            return

        if not isinstance(data, VariablesGroup):
            raise NotImplementedError(
                f"MaxPool: kernel_shape={kernel_shape} is not supported for a "
                "single-column input."
            )

        data = NumericVariablesGroup(data)
        cols = list(data.values())
        n = len(cols)

        # Global max pool: kernel covers all columns.
        if len(kernel_shape) == 1 and kernel_shape[0] == n:
            result: ibis.expr.types.NumericValue = cols[0]
            for col in cols[1:]:
                result = ibis.greatest(result, col)
            self.set_output(self._optimizer.fold_operation(result))
            return

        raise NotImplementedError(
            f"MaxPool: kernel_shape={kernel_shape} with {n} input column(s) is "
            "not supported in the SQL context.  Only kernel_shape=[1] (identity) "
            "or kernel_shape=[n_cols] (global max) are supported."
        )
