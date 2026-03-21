"""Implementation of the Clip operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup


class ClipTranslator(Translator):
    """Clips tensor values to a [min, max] range.

    Supports ONNX opset < 11 (min/max as node attributes) and
    opset >= 11 (min/max as optional initializer inputs).

    When min or max are absent (``None``), the corresponding bound is not applied.
    """

    def process(self) -> None:
        """Performs the translation and sets the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Clip.html
        data = self._variables.consume(self.inputs[0])

        # ONNX opset < 11: bounds are node attributes
        min_val: typing.Optional[float] = self._attributes.get("min", None)
        max_val: typing.Optional[float] = self._attributes.get("max", None)

        # ONNX opset >= 11: bounds are optional initializer inputs [data, min, max]
        if len(self.inputs) > 1:
            v = self._variables.get_initializer_value(self.inputs[1])
            if v is not None:
                min_val = float(v[0]) if isinstance(v, (list, tuple)) else float(v)
        if len(self.inputs) > 2:
            v = self._variables.get_initializer_value(self.inputs[2])
            if v is not None:
                max_val = float(v[0]) if isinstance(v, (list, tuple)) else float(v)

        def _clip(
            val: ibis.expr.types.NumericValue,
        ) -> ibis.expr.types.NumericValue:
            if min_val is not None:
                val = ibis.greatest(ibis.literal(min_val), val)
            if max_val is not None:
                val = ibis.least(ibis.literal(max_val), val)
            return val

        if isinstance(data, VariablesGroup):
            self.set_output(
                ValueVariablesGroup(
                    {k: _clip(typing.cast(ibis.expr.types.NumericValue, v)) for k, v in data.items()}
                )
            )
        else:
            self.set_output(_clip(typing.cast(ibis.expr.types.NumericValue, data)))
