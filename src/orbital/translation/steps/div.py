"""Defines the translation step for the Div operation."""

import typing

import ibis

from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup
from ._base_binary_elementwise import BinaryElementwiseTranslator


class DivTranslator(BinaryElementwiseTranslator):
    """Processes a Div node and updates the variables with the output expression.

    This class is responsible for handling the division operation in the
    translation process. It takes two inputs: the first operand and the second
    operand (divisor).

    The first operand can be a column group or a single column,
    while the second operand must be a constant value.

    When the second operand is a single value, all columns of the column
    group are divided for that value. If the second operand is instead
    a list, each column of the column group is divided for the corresponding
    value in the list.
    """

    def _op(
        self,
        a: ibis.expr.types.NumericValue,
        b: ibis.expr.types.NumericValue,
    ) -> ibis.expr.types.NumericValue:
        return a / b

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Div.html

        first_operand = self._variables.consume(self._inputs[0])
        # Try constant initializer; fall back to a variable for
        # element-wise variable÷variable division.
        second_arg = self._variables.consume(self._inputs[1])
        if second_arg is None or not isinstance(
            second_arg, (list, tuple, ibis.Expr, VariablesGroup)
        ):
            raise NotImplementedError(
                "Div: Second input (divisor) must be a constant list or a variable column/group."
            )

        if isinstance(second_arg, (list, tuple)):
            # -------- constant divisor path (original behaviour) --------
            if isinstance(first_operand, VariablesGroup):
                first_operand = NumericVariablesGroup(first_operand)
                struct_fields = list(first_operand.keys())
                for value in first_operand.values():
                    if not isinstance(value, ibis.expr.types.NumericValue):
                        raise ValueError(
                            "Div: The first operand must be a numeric value."
                        )

                first_operand = typing.cast(
                    dict[str, ibis.expr.types.NumericValue], first_operand
                )
                if len(second_arg) == 1:
                    divisor = second_arg[0]
                    if not isinstance(divisor, (int, float)):
                        raise ValueError(
                            "Div: The second operand must be a numeric value."
                        )
                    self.set_output(
                        ValueVariablesGroup(
                            {
                                field: (
                                    self._optimizer.fold_operation(
                                        first_operand[field] / ibis.literal(divisor)
                                    )
                                )
                                for field in struct_fields
                            }
                        )
                    )
                else:
                    if len(second_arg) != len(first_operand):
                        raise ValueError(
                            "The number of elements in the second operand must match the number of columns in the first operand."
                        )
                    self.set_output(
                        ValueVariablesGroup(
                            {
                                field: (
                                    self._optimizer.fold_operation(
                                        first_operand[field] / second_arg[i]
                                    )
                                )
                                for i, field in enumerate(struct_fields)
                            }
                        )
                    )
            else:
                if not isinstance(first_operand, ibis.expr.types.NumericValue):
                    raise ValueError("Div: The first operand must be a numeric value.")
                if len(second_arg) != 1:
                    raise ValueError(
                        "when first operand is a single column, second operand must contain only one value."
                    )

                first_operand = typing.cast(ibis.expr.types.NumericValue, first_operand)
                self.set_output(
                    self._optimizer.fold_operation(first_operand / second_arg[0])
                )

        elif isinstance(second_arg, (ibis.Expr, VariablesGroup)):
            # -------- variable÷variable path --------
            if isinstance(first_operand, VariablesGroup) and isinstance(
                second_arg, VariablesGroup
            ):
                f_keys = list(first_operand.keys())
                s_vals = list(second_arg.values())
                first_num = NumericVariablesGroup(first_operand)
                if len(s_vals) == 1:
                    # broadcast: divide each column by the single divisor column
                    divisor_expr = typing.cast(ibis.expr.types.NumericValue, s_vals[0])
                    self.set_output(
                        ValueVariablesGroup(
                            {
                                k: self._optimizer.fold_operation(
                                    first_num[k] / divisor_expr
                                )
                                for k in f_keys
                            }
                        )
                    )
                elif len(f_keys) != len(s_vals):
                    raise ValueError(
                        "Div: both variable operands must have the same number of columns "
                        f"({len(f_keys)} vs {len(s_vals)})"
                    )
                else:
                    second_num = NumericVariablesGroup(
                        {k: v for k, v in zip(f_keys, s_vals)}
                    )
                    self.set_output(
                        ValueVariablesGroup(
                            {
                                k: self._optimizer.fold_operation(
                                    first_num[k] / second_num[k]
                                )
                                for k in f_keys
                            }
                        )
                    )
            elif isinstance(first_operand, ibis.Expr) and isinstance(
                second_arg, ibis.Expr
            ):
                self.set_output(
                    self._optimizer.fold_operation(
                        typing.cast(ibis.expr.types.NumericValue, first_operand)
                        / typing.cast(ibis.expr.types.NumericValue, second_arg)
                    )
                )
            else:
                raise ValueError(
                    "Div: both operands must be the same type "
                    "(both a column group or both a single column)."
                )

        else:
            raise NotImplementedError(
                "Div: second input must be a constant list or a variable column/group."
            )
