"""Implementation of the GlobalMaxPool operator.

Reduces a ``VariablesGroup`` of C columns (channels / sequence positions) to a
single per-row maximum value — the SQL analogue of global max-pooling over the
spatial / sequence dimension.

For a single-column input the operation is the identity.

References
----------
https://onnx.ai/onnx/operators/onnx__GlobalMaxPool.html
"""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class GlobalMaxPoolTranslator(Translator):
    def process(self) -> None:
        data = self._variables.consume(self.inputs[0])

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            cols = list(data.values())
            if not cols:
                raise ValueError(
                    "GlobalMaxPool: input group must have at least one column."
                )
            result: ibis.expr.types.NumericValue = cols[0]
            for col in cols[1:]:
                result = ibis.greatest(result, col)
            self.set_output(self._optimizer.fold_operation(result))
        else:
            self.set_output(data)
