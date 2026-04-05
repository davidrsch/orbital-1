"""Stub for the Shape operator (ONNX opset 1+).

Shape returns the shape of the input tensor as a 1-D integer tensor.
Shape information is a compile-time / graph-level concept; in the orbital
tabular model every column is a scalar expression evaluated per row, so no
shape tensor can be produced at runtime.

Raises
------
NotImplementedError
    Always, because shape extraction is not representable as a per-row SQL
    expression in the orbital tabular execution model.

References
----------
https://onnx.ai/onnx/operators/onnx__Shape.html
"""

from ..translator import Translator


class ShapeTranslator(Translator):
    """Stub translator that always raises for the Shape operator."""

    def process(self) -> None:
        """Raise NotImplementedError: Shape is not supported."""
        raise NotImplementedError(
            "Shape is not supported by orbital: the Shape operator returns a "
            "tensor of dimension sizes, which is a graph-level metadata concept "
            "that cannot be expressed as a per-row SQL expression."
        )
