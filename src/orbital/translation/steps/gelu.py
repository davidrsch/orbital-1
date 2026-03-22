"""Implementation of the Gelu operator (ONNX opset 20+).

References
----------
https://onnx.ai/onnx/operators/onnx__Gelu.html
"""

import math
import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup
from .erf import _erf_approx
from .tanh import _tanh

_SQRT_2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


class GeluTranslator(Translator):
    def process(self) -> None:
        # approximate="tanh": 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x³)))
        # approximate="none": 0.5 * x * (1 + erf(x / sqrt(2)))
        data = self._variables.consume(self.inputs[0])
        approximate = str(self._attributes.get("approximate", "none")).lower()

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("Gelu: The input must be a numeric column.")

        def _gelu(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
            if approximate == "tanh":
                inner = ibis.literal(_SQRT_2_OVER_PI) * (
                    v + ibis.literal(0.044715) * v ** ibis.literal(3.0)
                )
                return ibis.literal(0.5) * v * (ibis.literal(1.0) + _tanh(inner))
            else:
                # "none" mode: use the erf polynomial approximation
                return (
                    ibis.literal(0.5)
                    * v
                    * (ibis.literal(1.0) + _erf_approx(v / ibis.literal(_SQRT_2)))
                )

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result: NumericVariablesGroup | ibis.expr.types.NumericValue = (
                NumericVariablesGroup({
                    k: self._optimizer.fold_operation(_gelu(v))
                    for k, v in data.items()
                })
            )
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(_gelu(data))

        self.set_output(result)
