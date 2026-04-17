"""Defines the translation step for the CumSum operation.

Only the column axis (axis=1) is supported for the tabular SQL context.
Supports the ``exclusive`` and ``reverse`` attributes.

References
----------
https://onnx.ai/onnx/operators/onnx__CumSum.html
"""

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup


class CumSumTranslator(Translator):
    """Translate the ONNX CumSum operator into prefix/suffix sum column expressions.

    Limitations:
    - Only ``axis=1`` (the column / feature axis) is supported; ``axis=0``
      (row-wise running sum) raises :class:`NotImplementedError` because SQL
      has no row ordering in a flat table context.
    - Input must be a dict (VariablesGroup) of columns.

    Supported combinations:
    - ``exclusive=0, reverse=0``: inclusive prefix sum  — out[k] = sum(x[0..k])
    - ``exclusive=1, reverse=0``: exclusive prefix sum  — out[k] = sum(x[0..k-1])
    - ``exclusive=0, reverse=1``: inclusive suffix sum  — out[k] = sum(x[k..n-1])
    - ``exclusive=1, reverse=1``: exclusive suffix sum  — out[k] = sum(x[k+1..n-1])
    """

    def process(self) -> None:
        """Perform the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__CumSum.html
        data = self._variables.consume(self.inputs[0])

        # axis is the second *input* (not an attribute).
        axis_raw = self._variables.consume(self.inputs[1])
        try:
            axis = int(axis_raw)
        except (TypeError, ValueError):
            try:
                axis = int(axis_raw.execute())
            except (AttributeError, TypeError, ValueError):
                axis = 1  # default to column axis

        exclusive = int(self._attributes.get("exclusive", 0))
        reverse = int(self._attributes.get("reverse", 0))

        if not isinstance(data, dict):
            raise NotImplementedError(
                "CumSumTranslator can only be applied to a group of columns"
            )

        if axis != 1:
            raise NotImplementedError(
                "CumSumTranslator only supports axis=1 (column / feature axis)"
            )

        keys = list(data.keys())
        n = len(keys)
        result: dict = {}

        if not reverse and not exclusive:
            # inclusive prefix sum: out[k] = x[0] + x[1] + ... + x[k]
            for k, key in enumerate(keys):
                expr = data[keys[0]]
                for j in range(1, k + 1):
                    expr = expr + data[keys[j]]
                result[key] = expr

        elif not reverse and exclusive:
            # exclusive prefix sum: out[k] = x[0] + ... + x[k-1]; out[0] = 0
            for k, key in enumerate(keys):
                if k == 0:
                    result[key] = ibis.literal(0)
                else:
                    expr = data[keys[0]]
                    for j in range(1, k):
                        expr = expr + data[keys[j]]
                    result[key] = expr

        elif reverse and not exclusive:
            # inclusive suffix sum: out[k] = x[k] + ... + x[n-1]
            for k, key in enumerate(keys):
                expr = data[keys[k]]
                for j in range(k + 1, n):
                    expr = expr + data[keys[j]]
                result[key] = expr

        else:
            # exclusive suffix sum: out[k] = x[k+1] + ... + x[n-1]; out[n-1] = 0
            for k, key in enumerate(keys):
                if k == n - 1:
                    result[key] = ibis.literal(0)
                else:
                    expr = data[keys[k + 1]]
                    for j in range(k + 2, n):
                        expr = expr + data[keys[j]]
                    result[key] = expr

        self.set_output(ValueVariablesGroup(result))
