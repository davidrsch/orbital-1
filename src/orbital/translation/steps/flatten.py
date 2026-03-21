"""Implementation of the Flatten operator."""

from ..translator import Translator


class FlattenTranslator(Translator):
    """Pass-through Flatten for SQL column groups.

    ``Flatten`` reshapes a tensor into shape ``(d_0…d_{axis-1}, d_axis…d_n)``.
    In SQL, data is already represented as independent scalar columns
    (effectively 1-D per row), so flattening at ``axis=1`` — the standard
    case for MLP preprocessing — is a no-op.
    """

    def process(self) -> None:
        """Performs the translation and sets the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Flatten.html
        data = self._variables.consume(self.inputs[0])

        axis = self._attributes.get("axis", 1)
        if axis != 1:
            raise NotImplementedError(
                f"Flatten: only axis=1 is supported in the SQL context, got axis={axis}"
            )

        self.set_output(data)
