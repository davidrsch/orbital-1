"""Defines the translation step for the GatherND operation.

For the tabular SQL context only ``batch_dims=0`` is supported and
``indices`` must be a single integer column whose values are 0-based
positions into the column list of ``data``.

References
----------
https://onnx.ai/onnx/operators/onnx__GatherND.html
"""

import ibis

from ..translator import Translator


class GatherNDTranslator(Translator):
    """Translate the ONNX GatherND operator via CASE-WHEN index dispatch.

    Limitations:
    - Only ``batch_dims=0`` is supported.
    - ``data`` must be a dict (VariablesGroup) of columns.
    - ``indices`` must be a single column of integer row-indices into
      the column list of ``data``.
    """

    def process(self) -> None:
        """Perform the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__GatherND.html
        data = self._variables.consume(self.inputs[0])
        indices = self._variables.consume(self.inputs[1])

        batch_dims = int(self._attributes.get("batch_dims", 0))

        if batch_dims != 0:
            raise NotImplementedError("GatherNDTranslator only supports batch_dims=0")

        if not isinstance(data, dict):
            raise NotImplementedError(
                "GatherNDTranslator can only be applied to a group of columns"
            )

        keys = list(data.keys())

        if len(keys) == 0:
            self.set_output(ibis.null())
            return

        if len(keys) == 1:
            self.set_output(data[keys[0]])
            return

        # Build a CASE-WHEN expression: for each column position i,
        # if indices == i then return data[keys[i]].
        *head_cases, (_, last_val) = [
            (indices == ibis.literal(i), data[key]) for i, key in enumerate(keys)
        ]
        result = ibis.cases(*head_cases, else_=last_val)

        self.set_output(result)
