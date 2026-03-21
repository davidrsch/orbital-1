"""Implementation of the Transpose operator."""

from ..translator import Translator


class TransposeTranslator(Translator):
    """Pass-through Transpose for SQL column groups.

    In SQL's row-per-record model every column is an independent scalar
    expression; there is no meaningful row-level axis to transpose.
    For the 2D weight-matrix shapes used in MLP operations, transposition
    is already handled structurally by ``Gemm``'s ``transB`` attribute, so
    at the variable level this node is a no-op.

    Only 2D permutations (``perm`` of length 2) are supported.
    """

    def process(self) -> None:
        """Performs the translation and sets the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Transpose.html
        data = self._variables.consume(self.inputs[0])

        perm = self._attributes.get("perm", None)
        if perm is not None and len(perm) > 2:
            raise NotImplementedError(
                f"Transpose: only 2D permutations are supported in SQL context, got perm={perm}"
            )

        self.set_output(data)
