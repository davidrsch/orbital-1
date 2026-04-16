"""Implementation of the Shape operator (ONNX opset 1+).

Shape returns the shape of the input tensor as a 1-D integer tensor.  In the
orbital tabular column model each "tensor" is a group of N scalar columns (one
per feature dimension).  The batch dimension (number of rows) cannot be
expressed as a static per-row SQL scalar.

Supported case
--------------
``start >= 1``: only feature dimensions are requested.  The number of input
feature columns is returned as an ``ibis.literal(N)`` integer expression.

Unsupported case
----------------
``start == 0``: the batch dimension is included.  Raises
:class:`NotImplementedError` because the row count is not accessible as a
per-row SQL scalar.

References
----------
https://onnx.ai/onnx/operators/onnx__Shape.html
"""

import ibis

from ..translator import Translator
from ..variables import VariablesGroup


class ShapeTranslator(Translator):
    """Translate the ONNX Shape operator to a static feature-count literal."""

    def process(self) -> None:
        """Translate the Shape node, writing result to the graph."""
        start = int(self._attributes.get("start", 0))

        if start == 0:
            raise NotImplementedError(
                "ShapeTranslator: batch dimension (dim 0) is not expressible as a "
                "static SQL value; use Shape with start=1 to extract feature "
                "dimensions only"
            )

        # Consume the input to mark it as used in the graph
        data = self._variables.consume(self.inputs[0])

        if isinstance(data, VariablesGroup):
            n_cols = len(data)
        elif isinstance(data, (list, tuple)):
            n_cols = len(data)
        else:
            n_cols = 1

        # Return the feature count as a static integer literal
        self.set_output(ibis.literal(n_cols))
