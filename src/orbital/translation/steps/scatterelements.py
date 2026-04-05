"""Stub for the ScatterElements operator (ONNX opset 11+).

ScatterElements performs scatter updates on individual elements of a tensor,
writing ``updates`` values to positions given by ``indices``.  This operation
requires per-row mutable tensor state that cannot be expressed as a stateless
SQL expression over flat tabular columns.

Raises
------
NotImplementedError
    Always, because the scatter operation is not representable as a pure
    per-row SQL expression in the orbital tabular execution model.

References
----------
https://onnx.ai/onnx/operators/onnx__ScatterElements.html
"""

from ..translator import Translator


class ScatterElementsTranslator(Translator):
    """Stub translator that always raises for the ScatterElements operator."""

    def process(self) -> None:
        """Raise NotImplementedError: ScatterElements is not supported."""
        raise NotImplementedError(
            "ScatterElements is not supported by orbital: the scatter operation "
            "requires mutable indexed-write semantics that cannot be expressed as "
            "a stateless per-row SQL expression."
        )
