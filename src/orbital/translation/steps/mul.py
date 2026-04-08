"""Translate an Mul operation to the equivalent query expression."""

import ibis

from ._base_binary_elementwise import BinaryElementwiseTranslator


class MulTranslator(BinaryElementwiseTranslator):
    """Processes an Mul node and updates the variables with the output expression.

    Given the node to translate, the variables and constants available for
    the translation context, generates a query expression that processes
    the input variables and produces a new output variable that computes
    based on the Mul operation.
    """

    def _op(
        self,
        a: ibis.expr.types.NumericValue,
        b: ibis.expr.types.NumericValue,
    ) -> ibis.expr.types.NumericValue:
        return a * b
