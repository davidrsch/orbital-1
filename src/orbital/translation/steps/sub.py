"""Implementation of the Sub operator."""

import typing

import ibis

from orbital.translation.variables import (
    NumericVariablesGroup,
    ValueVariablesGroup,
    VariablesGroup,
)

from ..translator import Translator


class SubTranslator(Translator):
    """Processes a Sub node and updates the variables with the output expression.

    Given the node to translate, the variables and constants available for
    the translation context, generates a query expression that processes
    the input variables and produces a new output variable that computes
    based on the Sub operation.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Sub.html
        assert len(self._inputs) == 2, "The Sub node must have exactly 2 inputs."

        first_raw = self._variables.consume(self._inputs[0])
        second_raw = self._variables.consume(self._inputs[1])

        # Case 1: variable - constant  (most common: Scaler/BN bias subtraction)
        if isinstance(first_raw, (ibis.Expr, VariablesGroup)) and isinstance(
            second_raw, (list, tuple)
        ):
            first_operand = first_raw
            sub_values = list(second_raw)
            type_check_var = first_operand
            if isinstance(type_check_var, VariablesGroup):
                type_check_var = next(iter(type_check_var.values()), None)
            if not isinstance(type_check_var, ibis.expr.types.NumericValue):
                raise ValueError("Sub: The first operand must be a numeric value.")
            if isinstance(first_operand, VariablesGroup):
                first_operand = NumericVariablesGroup(first_operand)
                struct_fields = list(first_operand.keys())
                assert len(sub_values) == len(struct_fields), (
                    f"The number of values in the initializer ({len(sub_values)}) must match the number of fields ({len(struct_fields)})"
                )
                self.set_output(
                    ValueVariablesGroup(
                        {
                            field: self._optimizer.fold_operation(
                                first_operand[field] - sub_values[i]
                            )
                            for i, field in enumerate(struct_fields)
                        }
                    )
                )
            else:
                if len(sub_values) != 1:
                    raise ValueError(
                        "When the first operand is a single column, the second operand must contain exactly 1 value"
                    )
                self.set_output(
                    self._optimizer.fold_operation(
                        typing.cast(ibis.expr.types.NumericValue, first_operand)
                        - sub_values[0]
                    )
                )

        # Case 2: constant - variable  (e.g., "1 - sigmoid(x)" in binary MLP classifier)
        elif isinstance(first_raw, (list, tuple)) and isinstance(second_raw, ibis.Expr):
            const_values = list(first_raw)
            if len(const_values) != 1:
                raise ValueError(
                    "Sub: When the first operand is a constant and the second is a single column, "
                    "the constant must contain exactly 1 value."
                )
            var_col = typing.cast(ibis.expr.types.NumericValue, second_raw)
            self.set_output(
                self._optimizer.fold_operation(ibis.literal(const_values[0]) - var_col)
            )

        # Case 3: scalar constant - variable (first_raw is a single scalar number)
        elif isinstance(first_raw, (int, float)) and isinstance(second_raw, ibis.Expr):
            var_col = typing.cast(ibis.expr.types.NumericValue, second_raw)
            self.set_output(
                self._optimizer.fold_operation(ibis.literal(float(first_raw)) - var_col)
            )

        # Case 4: variable - variable
        elif isinstance(first_raw, (ibis.Expr, VariablesGroup)) and isinstance(
            second_raw, (ibis.Expr, VariablesGroup)
        ):
            if isinstance(first_raw, VariablesGroup) and isinstance(
                second_raw, VariablesGroup
            ):
                f_keys = list(first_raw.keys())
                s_vals = list(second_raw.values())
                if len(f_keys) != len(s_vals):
                    raise ValueError(
                        f"Sub: both variable operands must have the same number of columns ({len(f_keys)} vs {len(s_vals)})"
                    )
                first_num = NumericVariablesGroup(first_raw)
                second_num = NumericVariablesGroup(
                    {k: v for k, v in zip(f_keys, s_vals)}
                )
                self.set_output(
                    ValueVariablesGroup(
                        {
                            k: self._optimizer.fold_operation(
                                first_num[k] - second_num[k]
                            )
                            for k in f_keys
                        }
                    )
                )
            elif isinstance(first_raw, ibis.Expr) and isinstance(second_raw, ibis.Expr):
                self.set_output(
                    self._optimizer.fold_operation(
                        typing.cast(ibis.expr.types.NumericValue, first_raw)
                        - typing.cast(ibis.expr.types.NumericValue, second_raw)
                    )
                )
            else:
                raise ValueError(
                    "Sub: both variable operands must be the same type "
                    "(both a column group or both a single column)."
                )

        elif not isinstance(
            first_raw, (ibis.Expr, VariablesGroup, list, tuple, int, float)
        ):
            raise ValueError("Sub: The first operand must be a numeric value.")
        elif not isinstance(
            second_raw, (ibis.Expr, VariablesGroup, list, tuple, int, float)
        ):
            raise ValueError("Sub: The second operand must be a numeric value.")
        else:
            raise NotImplementedError(
                "Sub: unsupported operand combination. Expected variable-constant, "
                "constant-variable, or variable-variable."
            )
