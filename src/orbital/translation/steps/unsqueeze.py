"""Translator for the Unsqueeze operation."""

from ..translator import Translator


class UnsqueezeTranslator(Translator):
    """Translate the ONNX Unsqueeze operator (identity in the SQL/tabular context)."""

    def process(self) -> None:
        """Translate the Unsqueeze node, writing result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
        # In a columnar representation, adding a size-1 dimension is a pass-through.
        data = self._variables.consume(self.inputs[0])
        self.set_output(data)
