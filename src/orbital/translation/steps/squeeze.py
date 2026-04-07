"""Translator for the Squeeze operation."""

# UnsqueezeTranslator lives in its own module; re-exported here for compatibility.
from .unsqueeze import UnsqueezeTranslator as UnsqueezeTranslator  # noqa: F401

from ..translator import Translator


class SqueezeTranslator(Translator):
    """Translate the ONNX Squeeze operator (identity in the SQL/tabular context)."""

    def process(self) -> None:
        """Translate the Squeeze node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__Squeeze.html
        # In a columnar representation, removing a size-1 dimension is a pass-through.
        data = self._variables.consume(self.inputs[0])
        self.set_output(data)

