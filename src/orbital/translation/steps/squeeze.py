"""Implementation of the Squeeze and Unsqueeze operators."""
from ..translator import Translator


class SqueezeTranslator(Translator):
    def process(self) -> None:
        # https://onnx.ai/onnx/operators/onnx__Squeeze.html
        # In a columnar representation, removing a size-1 dimension is a pass-through.
        data = self._variables.consume(self.inputs[0])
        self.set_output(data)


class UnsqueezeTranslator(Translator):
    def process(self) -> None:
        # https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
        # In a columnar representation, adding a size-1 dimension is a pass-through.
        data = self._variables.consume(self.inputs[0])
        self.set_output(data)
