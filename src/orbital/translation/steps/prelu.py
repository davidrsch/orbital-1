"""Implementation of the PRelu operator.

PRelu(x, slope) = x         if x >= 0
                = slope * x  otherwise

The *slope* tensor is the second input (``inputs[1]``) and must be a constant
initializer, NOT an attribute.

References
----------
https://onnx.ai/onnx/operators/onnx__PRelu.html
"""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class PreluTranslator(Translator):
    """Translate the ONNX PRelu operator: ``x if x >= 0 else slope*x``."""

    def process(self) -> None:
        """Translate the PRelu node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])
        slope_val = self._variables.get_initializer_value(self.inputs[1])

        if slope_val is None:
            raise ValueError(
                "PRelu: slope (inputs[1]) must be a constant initializer tensor."
            )
        if not isinstance(slope_val, (list, tuple)):
            slope_val = [slope_val]
        slopes = [float(s) for s in slope_val]

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("PRelu: The input must be a numeric column.")

        def _prelu_col(
            v: ibis.expr.types.NumericValue, slope: float
        ) -> ibis.expr.types.NumericValue:
            return ibis.cases(
                (v >= ibis.literal(0.0), v),
                else_=ibis.literal(slope) * v,
            )

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            keys = list(data.keys())
            cols = list(data.values())
            result: NumericVariablesGroup | ibis.expr.types.NumericValue = (
                NumericVariablesGroup(
                    {
                        keys[i]: self._optimizer.fold_operation(
                            _prelu_col(cols[i], slopes[i % len(slopes)])
                        )
                        for i in range(len(keys))
                    }
                )
            )
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(_prelu_col(data, slopes[0]))

        self.set_output(result)
