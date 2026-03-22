"""Implementation of the Dropout operator.

Dropout is a no-op at inference time: every element is passed through unchanged
regardless of the *ratio* attribute.  This translator therefore simply forwards
its input to its **first** output (the masked tensor), leaving the optional
second output (the mask) unset, which is the correct behaviour for inference.
"""

from ..translator import Translator


class DropoutTranslator(Translator):
    def process(self) -> None:
        # https://onnx.ai/onnx/operators/onnx__Dropout.html
        # At inference mode all inputs are retained, so this is identity.
        data = self._variables.consume(self.inputs[0])
        self.set_output(data, index=0)
