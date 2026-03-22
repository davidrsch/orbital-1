"""Shared base class for element-wise activation (unary) translators."""

import abc
import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class UnaryActivationTranslator(Translator):
    """Base class for element-wise activation translators (Relu, Sigmoid, Tanh …).

    Subclasses only need to implement :meth:`_apply`.  The base class handles:
    - consuming the single input variable
    - type-checking (must be numeric)
    - mapping over a ``VariablesGroup`` or a bare column
    - wrapping each output in ``_optimizer.fold_operation``
    """

    @abc.abstractmethod
    def _apply(
        self, v: ibis.expr.types.NumericValue
    ) -> ibis.expr.types.NumericValue:
        """Apply the activation to a single ibis numeric expression."""

    def process(self) -> None:
        """Translates the activation node and writes the result to the graph."""
        data = self._variables.consume(self.inputs[0])

        # ---- type-check --------------------------------------------------- #
        type_check = data
        if isinstance(type_check, VariablesGroup):
            type_check = next(iter(type_check.values()), None)
        if not isinstance(type_check, ibis.expr.types.NumericValue):
            raise ValueError(
                f"{self.operation}: input must be a numeric column "
                "or a column group of numerics."
            )

        # ---- dispatch: group vs. scalar ------------------------------------ #
        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            result: ibis.expr.types.NumericValue | NumericVariablesGroup = (
                NumericVariablesGroup(
                    {
                        k: self._optimizer.fold_operation(self._apply(v))
                        for k, v in data.items()
                    }
                )
            )
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
            result = self._optimizer.fold_operation(self._apply(data))

        self.set_output(result)
