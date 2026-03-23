"""Implementation of the Pow operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class PowTranslator(Translator):
    """Translate the ONNX Pow operator: element-wise power ``x ** y``."""

    def process(self) -> None:
        """Translate the Pow node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__Pow.html
        first_operand = self._variables.consume(self.inputs[0])

        type_check = first_operand
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError("Pow: The base operand must be a numeric column.")

        # Try to get exponent from initializer
        exp_init = self._variables.get_initializer_value(self.inputs[1])
        if exp_init is not None:
            if isinstance(exp_init, (list, tuple)):
                if len(exp_init) != 1:
                    raise NotImplementedError(
                        "Pow: multi-value exponent initializers are not supported."
                    )
                exp_val = float(exp_init[0])
            else:
                exp_val = float(exp_init)

            def _op(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
                return v ** ibis.literal(exp_val)
        else:
            # Exponent is a computed variable (e.g. output of a previous node)
            second_operand = self._variables.consume(self.inputs[1])
            if not isinstance(second_operand, ibis.expr.types.NumericValue):
                raise NotImplementedError(
                    "Pow: exponent must be a scalar numeric expression."
                )
            _exponent = second_operand

            def _op(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:  # type: ignore[misc]
                return v**_exponent

        if isinstance(first_operand, VariablesGroup):
            first_operand = NumericVariablesGroup(first_operand)
            result: ibis.expr.types.NumericValue | NumericVariablesGroup = (
                NumericVariablesGroup(
                    {
                        k: self._optimizer.fold_operation(_op(v))
                        for k, v in first_operand.items()
                    }
                )
            )
        else:
            first_operand = typing.cast(ibis.expr.types.NumericValue, first_operand)
            result = self._optimizer.fold_operation(_op(first_operand))

        self.set_output(result)
