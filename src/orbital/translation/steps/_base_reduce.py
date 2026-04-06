"""Shared base class for ONNX Reduce* axis-reduction translators."""

from ..translator import Translator


class _ReduceAxisTranslator(Translator):
    """Base class for ONNX Reduce* operators that reduce over the feature axis.

    Provides :meth:`_extract_reduce_args` which handles the shared axes-validation
    and opset-18 ``noop_with_empty_axes`` guard present in all Reduce translators.
    """

    def _extract_reduce_args(self, op_name: str):
        """Consume input, validate axes, and handle the opset-18 noop guard.

        Parameters
        ----------
        op_name:
            Human-readable operator name used in error messages (e.g. ``"ReduceSum"``).

        Returns
        -------
        tuple[data, int] | None
            ``(data, keepdims)`` when reduction should proceed — ``data`` is the raw
            consumed value (may be a :class:`VariablesGroup` or an ibis expression).
            Returns ``None`` when the output has already been written via a noop
            pass-through and the caller should return immediately.
        """
        data = self._variables.consume(self.inputs[0])

        # Axes can be an attribute (opset < 18) or a second input (opset >= 18).
        axes = self._attributes.get("axes", None)
        if axes is not None:
            axes = list(axes)

        if len(self.inputs) > 1:
            axes_val = self._variables.get_initializer_value(self.inputs[1])
            if axes_val is not None:
                axes = [int(a) for a in axes_val]

        if axes is not None and not all(a in (-1, 1) for a in axes):
            raise NotImplementedError(
                f"{op_name}: only axis=-1 or axis=1 (feature axis) is supported, got {axes}"
            )

        keepdims = int(self._attributes.get("keepdims", 1))
        noop_with_empty_axes = int(self._attributes.get("noop_with_empty_axes", 0))

        # opset 18+: when axes input is empty and noop_with_empty_axes=1, pass through.
        if noop_with_empty_axes == 1 and (axes is None or len(axes) == 0):
            self.set_output(data)
            return None

        return data, keepdims
