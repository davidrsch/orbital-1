"""Implementation of the ONNX Slice operator for extracting column sub-sequences."""

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup


class SliceTranslator(Translator):
    """Translate the ONNX Slice operator for extracting a contiguous sub-sequence.

    In the tabular 1-D context the input is a :class:`VariablesGroup` whose
    columns represent a flat feature vector.  ``Slice`` extracts the columns
    in the half-open range ``[start, end)`` along the specified axis with an
    optional step.

    Supported configurations
    ------------------------
    - Rank-1 input or rank-2 with batch axis untouched.
    - ``starts``, ``ends``, ``axes`` (optional), and ``steps`` (optional) must
      all be constant initializers.
    - Only a single slice range is supported (``len(starts) == 1``).
    - ``axes`` must be 0 for rank-1 or 1 for rank-2 (feature axis).
    - ``steps`` must be 1 or -1.  Step -1 reverses the selected range.
    - Negative indices are resolved relative to the sequence length.

    References
    ----------
    https://onnx.ai/onnx/operators/onnx__Slice.html
    """

    # Sentinel used for the ONNX "INT_MAX" end value.
    _INT_MAX = 2**31 - 1

    def process(self) -> None:
        """Translate the Slice node, writing the sliced group to the graph."""
        # ── Consume inputs (opset ≥ 10: starts/ends/axes/steps as inputs) ─
        data = self._variables.consume(self.inputs[0])
        if isinstance(data, VariablesGroup):
            in_cols = list(data.values())
        else:
            in_cols = [data]
        n = len(in_cols)

        starts_raw = self._variables.get_initializer_value(self.inputs[1])
        ends_raw = self._variables.get_initializer_value(self.inputs[2])
        if starts_raw is None or ends_raw is None:
            raise NotImplementedError(
                "Slice: 'starts' and 'ends' must be constant initializers."
            )
        starts = [int(v) for v in starts_raw]
        ends = [int(v) for v in ends_raw]

        axes_raw = None
        if len(self.inputs) > 3 and self.inputs[3]:
            axes_raw = self._variables.get_initializer_value(self.inputs[3])
        axes = [int(v) for v in axes_raw] if axes_raw is not None else list(range(len(starts)))

        steps_raw = None
        if len(self.inputs) > 4 and self.inputs[4]:
            steps_raw = self._variables.get_initializer_value(self.inputs[4])
        steps = [int(v) for v in steps_raw] if steps_raw is not None else [1] * len(starts)

        if len(starts) != 1:
            raise NotImplementedError(
                "Slice: only a single slice range (len(starts)==1) is supported."
            )

        axis = axes[0]
        start = starts[0]
        end = ends[0]
        step = steps[0]

        # For rank-2, the feature axis is 1; batch axis cannot be sliced.
        if axis not in (0, 1):
            raise NotImplementedError(
                f"Slice: axis={axis} is not supported; only 0 (rank-1) or 1 (rank-2 features)."
            )

        if step not in (1, -1):
            raise NotImplementedError(
                f"Slice: step={step} is not supported; only step=1 or step=-1."
            )

        # Resolve negative indices and INT_MAX sentinel.
        if start < 0:
            start = start + n
        if end < 0:
            end = end + n
        end = min(end, n)  # clamp INT_MAX or out-of-bound ends

        selected = in_cols[start:end:step]

        out: dict[str, object] = {}
        for i, col in enumerate(selected):
            out[f"slice_{i}"] = col

        self.set_output(ValueVariablesGroup(out))
