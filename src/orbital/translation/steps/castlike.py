"""Translator for the CastLike operation."""

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup


class CastLikeTranslator(Translator):
    """Processes a CastLike node and updates the variables with the output expression.

    CastLike operation is used to convert a variable from one type to
    the same type of another variable, thus uniforming the two
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__CastLike.html

        # Cast a variable to have the same type of another variable.
        # For the moment provide a very minimal implementation,
        # in most cases this is used to cast concatenated features to the same type
        # of another feature.
        expr = self._variables.consume(self.inputs[0])
        like_expr = self._variables.consume(self.inputs[1])

        # Assert that the first input is a dict (multiple concatenated columns).
        if not isinstance(expr, VariablesGroup):
            # Known limitation: single-variable (non-group) CastLike is not yet
            # implemented. All observed sklearn2onnx models use CastLike on column
            # groups only. Single-variable support can be added when a concrete
            # model requires it.
            raise NotImplementedError(
                "CastLike currently only supports casting a group of columns."
            )

        # Assert that the second input is a single expression.
        if isinstance(like_expr, VariablesGroup):
            raise NotImplementedError(
                "CastLike currently only supports casting to a single column type, not a group."
            )

        if not isinstance(like_expr, ibis.Value):
            raise ValueError(
                f"CastLike: expected a single column. Got {type(like_expr)}"
            )

        # Get the target type from the second input.
        target_type: ibis.DataType = like_expr.type()

        # Now cast each field in the dictionary to the target type.
        casted = ValueVariablesGroup(
            {
                key: self._optimizer.fold_cast(expr.as_value(key).cast(target_type))
                for key in expr
            }
        )
        self.set_output(casted)
