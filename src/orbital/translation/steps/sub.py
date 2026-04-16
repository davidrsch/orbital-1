"""Implementation of the Sub operator."""

import typing

import ibis

from ..variables import VariablesGroup
from ._base_binary_elementwise import BinaryElementwiseTranslator


class SubTranslator(BinaryElementwiseTranslator):
    """Processes a Sub node and updates the variables with the output expression.

    Given the node to translate, the variables and constants available for
    the translation context, generates a query expression that processes
    the input variables and produces a new output variable that computes
    based on the Sub operation.
    """

    def _op(
        self,
        a: ibis.expr.types.NumericValue,
        b: ibis.expr.types.NumericValue,
    ) -> ibis.expr.types.NumericValue:
        return a - b

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Sub.html
        if len(self._inputs) != 2:
            raise ValueError(
                f"Sub: expected exactly 2 inputs, got {len(self._inputs)}."
            )

        first_raw = self._variables.consume(self._inputs[0])
        second_raw = self._variables.consume(self._inputs[1])

        # Case 1: variable - constant  (most common: Scaler/BN bias subtraction)
        if isinstance(first_raw, (ibis.Expr, VariablesGroup)) and isinstance(
            second_raw, (list, tuple)
        ):
            self._process_var_const(first_raw, list(second_raw))

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
            self._process_var_var(first_raw, second_raw)

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
