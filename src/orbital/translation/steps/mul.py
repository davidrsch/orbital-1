"""Translate an Mul operation to the equivalent query expression."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


class MulTranslator(Translator):
    """Processes an Mul node and updates the variables with the output expression.

    Given the node to translate, the variables and constants available for
    the translation context, generates a query expression that processes
    the input variables and produces a new output variable that computes
    based on the Mul operation.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Mul.html

        first_operand = self._variables.consume(self._inputs[0])
        # Try constant initializer first; fall back to a variable for
        # element-wise variable×variable multiplication.
        second_raw = self._variables.consume(self._inputs[1])

        type_check_var = first_operand
        if isinstance(type_check_var, VariablesGroup):
            type_check_var = next(iter(type_check_var.values()), None)
        if not isinstance(type_check_var, ibis.expr.types.NumericValue):
            raise ValueError("Mul: The first operand must be a numeric value.")

        if isinstance(second_raw, (list, tuple)):
            # -------- constant-multiplier path (original behaviour) --------
            add_values = list(second_raw)
            if isinstance(first_operand, VariablesGroup):
                first_operand = NumericVariablesGroup(first_operand)
                struct_fields = list(first_operand.keys())
                if len(add_values) != len(struct_fields):
                    # TODO: Implement dividing by a single value,
                    #       see Div implementation.
                    raise ValueError(
                        "When the first operand is a group of columns, the second operand must contain the same number of values"
                    )
                self.set_output(
                    ValueVariablesGroup(
                        {
                            field: (
                                self._optimizer.fold_operation(
                                    first_operand[field] * add_values[i]
                                )
                            )
                            for i, field in enumerate(struct_fields)
                        }
                    )
                )
            else:
                if len(add_values) != 1:
                    raise ValueError(
                        "When the first operand is a single column, the second operand must contain exactly 1 value"
                    )
                first_operand = typing.cast(ibis.expr.types.NumericValue, first_operand)
                self.set_output(
                    self._optimizer.fold_operation(first_operand * add_values[0])
                )

        elif isinstance(second_raw, (ibis.Expr, VariablesGroup)):
            # -------- variable×variable path --------
            if isinstance(first_operand, VariablesGroup) and isinstance(
                second_raw, VariablesGroup
            ):
                f_keys = list(first_operand.keys())
                s_vals = list(second_raw.values())
                if len(f_keys) != len(s_vals):
                    raise ValueError(
                        "Mul: both variable operands must have the same number of columns "
                        f"({len(f_keys)} vs {len(s_vals)})"
                    )
                first_num = NumericVariablesGroup(first_operand)
                second_num = NumericVariablesGroup(
                    {k: v for k, v in zip(f_keys, s_vals)}
                )
                self.set_output(
                    ValueVariablesGroup(
                        {
                            k: self._optimizer.fold_operation(
                                first_num[k] * second_num[k]
                            )
                            for k in f_keys
                        }
                    )
                )
            elif isinstance(first_operand, ibis.Expr) and isinstance(
                second_raw, ibis.Expr
            ):
                self.set_output(
                    self._optimizer.fold_operation(
                        typing.cast(ibis.expr.types.NumericValue, first_operand)
                        * typing.cast(ibis.expr.types.NumericValue, second_raw)
                    )
                )
            else:
                raise ValueError(
                    "Mul: both operands must be the same type "
                    "(both a column group or both a single column)."
                )

        else:
            raise NotImplementedError(
                "Mul: second input must be a constant list or a variable column/group."
            )
