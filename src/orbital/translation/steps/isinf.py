"""Implementation of the IsInf operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class IsInfTranslator(Translator):
    """Translate the ONNX IsInf operator: element-wise boolean infinity check."""

    def process(self) -> None:
        """Translate the IsInf node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__IsInf.html
        # ibis has no direct .isinf(), so we compare against literal infinities.
        detect_positive = int(self._attributes.get("detect_positive", 1))
        detect_negative = int(self._attributes.get("detect_negative", 1))

        data = self._variables.consume(self.inputs[0])

        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("IsInf: The input must be a numeric column.")

        pos_inf = ibis.literal(float("inf"))
        neg_inf = ibis.literal(float("-inf"))

        def _isinf(v: ibis.expr.types.NumericValue) -> ibis.expr.types.BooleanValue:
            if detect_positive and detect_negative:
                return (v == pos_inf) | (v == neg_inf)
            elif detect_positive:
                return v == pos_inf
            elif detect_negative:
                return v == neg_inf
            else:
                return ibis.literal(False)

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result = NumericVariablesGroup(
                {k: self._optimizer.fold_operation(_isinf(v)) for k, v in data.items()}
            )
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(_isinf(data))

        self.set_output(result)
