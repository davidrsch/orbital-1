"""Translator for the FeatureVectorizer operation (ai.onnx.ml)."""

import typing

from ..translator import Translator
from .concat import ConcatTranslator


class FeatureVectorizerTranslator(Translator):
    """Concatenate multiple columns into a single group of columns.

    This is similar to Concat, but it is a simplified version
    that always only acts on columns, and does not support
    concatenating over rows. While Concat can in theory
    support rows concatenation, even though orbital doesn't implement it.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx_aionnxml_FeatureVectorizer.html

        # We can support this by doing the same as Concat,
        # in most cases it's sufficient
        ninputdimensions = typing.cast(list[int], self._attributes["inputdimensions"])

        if len(ninputdimensions) != len(self._inputs):
            raise ValueError(
                "Number of input dimensions should be equal to number of inputs."
            )

        # Validate that dimensions are actually correct,
        # as inputdimensions is meant to provide the number of columns of each variable
        for input_idx, colname in enumerate(self.inputs):
            dimensions = ninputdimensions[input_idx]
            feature = self._variables.peek_variable(colname)
            if isinstance(feature, dict):
                if len(feature) != dimensions:
                    raise ValueError(
                        f"Number of columns in input {colname} should be equal to the number of dimensions, got {len(feature)} != {dimensions}"
                    )
            else:
                if dimensions != 1:
                    raise ValueError(
                        f"When merging over individual columns, the dimension should be 1, got {dimensions} for {colname}"
                    )

        self.set_output(ConcatTranslator._concatenate_columns(self))
