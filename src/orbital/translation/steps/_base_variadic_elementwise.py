"""Shared base class for variadic element-wise ONNX operators.

Variadic element-wise operators (Sum, Max, Min, Mean) accept N input tensors of
the same shape and produce a single output by applying an operation across all
inputs element-wise.

Unlike the binary element-wise operators in ``_base_binary_elementwise.py``,
these operators do not have a broadcast path — all inputs must be compatible
(same column count when VariablesGroups are involved).

References
----------
https://onnx.ai/onnx/operators/onnx__Sum.html
"""

import abc

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


class _VariadicElementwiseTranslator(Translator):
    """Base for variadic element-wise ONNX operators: Sum, Max, Min, Mean.

    Subclasses implement :meth:`_combine` to fold two ibis expressions into
    one (e.g. addition, ``ibis.greatest``, ``ibis.least``).  The base class
    handles consuming N inputs and dispatching over the single-expression vs
    VariablesGroup cases.
    """

    @abc.abstractmethod
    def _combine(
        self,
        a: ibis.expr.types.NumericValue,
        b: ibis.expr.types.NumericValue,
    ) -> ibis.expr.types.NumericValue:
        """Fold two numeric expressions into one using the operator's semantics."""

    def _post_process(
        self,
        expr: ibis.expr.types.NumericValue,
        n_inputs: int,
    ) -> ibis.expr.types.NumericValue:
        """Optional post-processing hook (used by Mean to divide by N)."""
        return expr

    def process(self) -> None:
        """Translate the variadic node, writing result to the graph."""
        raw_inputs = [self._variables.consume(name) for name in self.inputs]
        if not raw_inputs:
            raise ValueError(f"{self.operation}: at least one input is required.")

        first = raw_inputs[0]

        if isinstance(first, VariablesGroup):
            # All inputs must be VariablesGroups with matching column counts.
            groups = [list(NumericVariablesGroup(x).values()) for x in raw_inputs]
            n_cols = len(groups[0])
            for i, g in enumerate(groups[1:], 1):
                if len(g) != n_cols:
                    raise ValueError(
                        f"{self.operation}: input {i} has {len(g)} columns but "
                        f"input 0 has {n_cols}; all inputs must have the same shape."
                    )
            result = ValueVariablesGroup()
            ref_keys = list(NumericVariablesGroup(first).keys())
            for col_i in range(n_cols):
                col_expr: ibis.expr.types.NumericValue = groups[0][col_i]
                for g in groups[1:]:
                    col_expr = self._combine(col_expr, g[col_i])
                col_expr = self._post_process(col_expr, len(raw_inputs))
                result[ref_keys[col_i]] = self._optimizer.fold_operation(col_expr)
            self.set_output(result)

        else:
            # All inputs are bare ibis expressions.
            acc: ibis.expr.types.NumericValue = first  # type: ignore[assignment]
            for inp in raw_inputs[1:]:
                acc = self._combine(acc, inp)  # type: ignore[arg-type]
            acc = self._post_process(acc, len(raw_inputs))
            self.set_output(self._optimizer.fold_operation(acc))
