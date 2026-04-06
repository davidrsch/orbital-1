"""Translator for the Concat operation."""

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup


class ConcatTranslator(Translator):
    """Concatenate multiple columns into a single group of columns.

    In tensor terms, this is meant to create a new tensor by concatenating
    the inputs along a given axis. In most cases, this is used to
    concatenate multiple features into a single one, thus its purpose
    is usually to create a column group from separate columns.

    This means that the most common use case is axis=1,
    which means concatenating over the columns (by virtue of
    column/rows in tensors being flipped over column groups),
    and thus only axis=1 case is supported.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Concat.html

        # Currently only support concatenating over columns,
        # we can't concatenate rows.
        if self._attributes["axis"] not in (1, -1):
            # -1 means last axis, which for 2D entities is equal axis=1
            raise NotImplementedError(
                "Concat currently only supports concatenating over columns (axis=1 or -1)."
            )
        self.set_output(self._concatenate_columns(self))

    @classmethod
    def _concatenate_columns(cls, translator: Translator) -> VariablesGroup:
        """Implement actual operation of concatenating columns.

        This is used by both Concat and FeatureVectorizer translators,
        as they both need to concatenate columns.
        """
        result = ValueVariablesGroup()

        for col in translator.inputs:
            feature = translator._variables.consume(col)
            if isinstance(feature, dict):
                # When the feature is a dictionary,  it means that it was previously
                # concatenated with other features. In pure ONNX terms it would be
                # a tensor, so when we concatenate it we should just merge all the values
                # like we would do when concatenating two tensors.
                for key in feature:
                    varname = col + "." + key
                    result[varname] = feature[key]
            elif isinstance(feature, ibis.Expr):
                result[col] = feature
            else:
                raise ValueError(
                    f"Concat: expected a column group or a single column. Got {type(feature)}"
                )

        return result
