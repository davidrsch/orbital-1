"""Implementation of the Softmax operator."""

import typing

import ibis

from ..transformations import apply_post_transform
from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class SoftmaxTranslator(Translator):
    """Processes a Softmax node and updates the variables with the output expression.

    The operation computes the normalized exponential of the input::

        Softmax(x)_i = Exp(x_i - max(x)) / Sum(Exp(x_j - max(x)))

    The numerically-stable form (subtract max before exponentiation) is
    implemented inside ``apply_post_transform(data, "SOFTMAX")``; see
    ``orbital.translation.transformations.apply_post_transform`` for the
    exact ibis expression tree.

    Currently the Softmax operation is supported only for axis=-1 or axis=1,
    which means for a column group that the softmax is computed
    independently for each row over all columns in the group.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Softmax.html
        data = self._variables.consume(self.inputs[0])
        if not isinstance(data, (ibis.expr.types.NumericValue, dict)):
            raise ValueError(
                "Softmax: The first operand must be a numeric column or a column group of numerics."
            )

        axis = self._attributes.get("axis", -1)
        if axis not in (-1, 1):
            raise ValueError(
                "SoftmaxTranslator supports only axis=-1 or axis=1 for group of columns"
            )

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
        self.set_output(apply_post_transform(data, "SOFTMAX"))
