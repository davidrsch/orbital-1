"""Shared base class for binary element-wise translators."""

import abc
import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


class BinaryElementwiseTranslator(Translator):
    """Base class for element-wise binary translators (Add, Sub, Mul, Div …).

    Subclasses only need to implement :meth:`_op`.  The base class handles:
    - consuming both input variables
    - type-checking the first operand (must be numeric)
    - dispatching over the two top-level cases:

      - *variable OP constant-list*  (``VariablesGroup`` or bare column)
      - *variable OP variable*       (``VariablesGroup`` or bare column)

    - wrapping each output in ``_optimizer.fold_operation``

    Operators with additional cases (e.g. Sub's *constant OP variable* paths,
    or Div's broadcast behaviour) should override :meth:`process` and call the
    protected helpers :meth:`_process_var_const` and :meth:`_process_var_var`
    for the shared paths.
    """

    @abc.abstractmethod
    def _op(
        self,
        a: ibis.expr.types.NumericValue,
        b: ibis.expr.types.NumericValue,
    ) -> ibis.expr.types.NumericValue:
        """Apply the binary operation to two numeric expressions."""

    def process(self) -> None:
        """Translates the node and writes the result to the graph."""
        first_raw = self._variables.consume(self._inputs[0])
        second_raw = self._variables.consume(self._inputs[1])

        if isinstance(second_raw, (list, tuple)):
            self._process_var_const(first_raw, list(second_raw))
        elif isinstance(second_raw, (ibis.Expr, VariablesGroup)):
            self._process_var_var(first_raw, second_raw)
        else:
            raise NotImplementedError(
                f"{self.operation}: second input must be a constant list or a variable"
                " column/group."
            )

    def _process_var_const(
        self,
        first_operand: ibis.Expr | VariablesGroup,
        const_values: list,
    ) -> None:
        """Handle the *variable OP constant-list* case."""
        type_check_var = first_operand
        if isinstance(type_check_var, VariablesGroup):
            type_check_var = next(iter(type_check_var.values()), None)
        if not isinstance(type_check_var, ibis.expr.types.NumericValue):
            raise ValueError(
                f"{self.operation}: The first operand must be a numeric value."
            )

        if isinstance(first_operand, VariablesGroup):
            first_operand = NumericVariablesGroup(first_operand)
            struct_fields = list(first_operand.keys())
            if len(const_values) != len(struct_fields):
                raise ValueError(
                    f"{self.operation}: The variable group has {len(struct_fields)} "
                    f"columns but the constant does not have the same number of values "
                    f"({len(const_values)} provided)."
                )
            self.set_output(
                ValueVariablesGroup(
                    {
                        field: self._optimizer.fold_operation(
                            self._op(first_operand[field], const_values[i])
                        )
                        for i, field in enumerate(struct_fields)
                    }
                )
            )
        else:
            if len(const_values) != 1:
                raise ValueError(
                    f"{self.operation}: When the first operand is a single column, "
                    "the second operand must contain exactly 1 value."
                )
            first_operand = typing.cast(ibis.expr.types.NumericValue, first_operand)
            self.set_output(
                self._optimizer.fold_operation(self._op(first_operand, const_values[0]))
            )

    def _process_var_var(
        self,
        first_operand: ibis.Expr | VariablesGroup,
        second_operand: ibis.Expr | VariablesGroup,
    ) -> None:
        """Handle the *variable OP variable* case."""
        if isinstance(first_operand, VariablesGroup) and isinstance(
            second_operand, VariablesGroup
        ):
            f_keys = list(first_operand.keys())
            s_vals = list(second_operand.values())
            if len(f_keys) != len(s_vals):
                raise ValueError(
                    f"{self.operation}: both variable operands must have the same"
                    f" number of columns ({len(f_keys)} vs {len(s_vals)})"
                )
            first_num = NumericVariablesGroup(first_operand)
            second_num = NumericVariablesGroup({k: v for k, v in zip(f_keys, s_vals)})
            self.set_output(
                ValueVariablesGroup(
                    {
                        k: self._optimizer.fold_operation(
                            self._op(first_num[k], second_num[k])
                        )
                        for k in f_keys
                    }
                )
            )
        elif isinstance(first_operand, ibis.Expr) and isinstance(
            second_operand, ibis.Expr
        ):
            self.set_output(
                self._optimizer.fold_operation(
                    self._op(
                        typing.cast(ibis.expr.types.NumericValue, first_operand),
                        typing.cast(ibis.expr.types.NumericValue, second_operand),
                    )
                )
            )
        else:
            raise ValueError(
                f"{self.operation}: both operands must be the same type "
                "(both a column group or both a single column)."
            )
