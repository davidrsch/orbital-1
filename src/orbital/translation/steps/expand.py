"""Defines the translation step for the Expand operation.

ONNX Expand broadcasts a tensor to a given shape.  For the tabular SQL
context only a scalar-to-N-columns expansion is supported: a single
input column is replicated N times to produce a dict with N entries.

References
----------
https://onnx.ai/onnx/operators/onnx__Expand.html
"""

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup


class ExpandTranslator(Translator):
    """Translate the ONNX Expand operator for the scalar → N-column case.

    Limitations:
    - ``input`` must be a single column expression (not a dict).
    - ``shape`` must be a scalar integer constant or a single-element
      dict/VariablesGroup whose value evaluates to an integer.
    - Only scalar-to-N-column expansion is supported; tensor broadcast
      of arbitrary shape raises :class:`NotImplementedError`.
    """

    def process(self) -> None:
        """Perform the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Expand.html
        input_val = self._variables.consume(self.inputs[0])
        shape_val = self._variables.consume(self.inputs[1])

        # Resolve input column.
        if isinstance(input_val, dict):
            col_list = list(input_val.values())
            if len(col_list) != 1:
                raise NotImplementedError(
                    "ExpandTranslator only supports a single-column input"
                )
            col = col_list[0]
        elif isinstance(input_val, ibis.Expr):
            col = input_val
        else:
            col = ibis.literal(input_val)

        # Resolve N (number of output columns).
        if isinstance(shape_val, dict):
            shape_list = list(shape_val.values())
            if len(shape_list) != 1:
                raise NotImplementedError(
                    "ExpandTranslator only supports a scalar shape constant"
                )
            n_raw = shape_list[0]
            try:
                n = int(n_raw)
            except (TypeError, ValueError):
                try:
                    n = int(n_raw.execute())
                except Exception:
                    raise NotImplementedError(
                        "ExpandTranslator could not resolve shape to a scalar integer"
                    )
        elif isinstance(shape_val, ibis.Expr):
            try:
                n = int(shape_val.op().value)
            except Exception:
                try:
                    n = int(shape_val.execute())
                except Exception:
                    raise NotImplementedError(
                        "ExpandTranslator could not resolve shape to a scalar integer"
                    )
        else:
            try:
                n = int(shape_val)
            except (TypeError, ValueError):
                raise NotImplementedError(
                    "ExpandTranslator only supports a scalar integer shape"
                )

        if n <= 0:
            raise ValueError(f"ExpandTranslator: shape N must be positive, got {n}")

        # Replicate the single column N times, keying by 0-based position.
        result = ValueVariablesGroup({i: col for i in range(n)})
        self.set_output(result)
