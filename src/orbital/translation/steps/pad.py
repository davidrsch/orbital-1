"""Implementation of the ONNX Pad operator for 1-D tabular sequences."""

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup


class PadTranslator(Translator):
    """Translate the ONNX Pad operator for constant-padding of feature sequences.

    In the tabular 1-D context the input is a :class:`VariablesGroup` whose
    columns are a flat ``[T × C]`` or ``[C]`` feature vector.  Padding inserts
    columns with the constant value (default 0) at the start and end of the
    feature axis.

    Supported configurations
    ------------------------
    - ``mode="constant"`` (the default) only.
    - ``pads`` supplied either as a node attribute (opset < 11) or as the
      second initializer input (opset ≥ 11).
    - Rank-1 tensors: ``pads = [p_begin, p_end]``.
    - Rank-2 tensors ``[batch, features]``:
      ``pads = [0, f_begin, 0, f_end]`` — batch dimension must be unpadded.
    - Higher ranks raise :class:`NotImplementedError`.

    References
    ----------
    https://onnx.ai/onnx/operators/onnx__Pad.html
    """

    def process(self) -> None:
        """Translate the Pad node, writing the padded group to the graph."""
        mode = str(self._attributes.get("mode", "constant"))
        if mode != "constant":
            raise NotImplementedError(
                f"Pad: mode='{mode}' is not supported; only 'constant' mode is."
            )

        # ── Read pads ──────────────────────────────────────────────────────
        # Opset < 11: pads is a node attribute.
        # Opset >= 11: pads is inputs[1] (initializer tensor).
        pads_raw = self._attributes.get("pads")
        if pads_raw is None and len(self.inputs) > 1 and self.inputs[1]:
            pads_raw = self._variables.get_initializer_value(self.inputs[1])
        if pads_raw is None:
            raise ValueError("Pad: 'pads' attribute / initializer not found.")
        pads = [int(p) for p in pads_raw]

        ndim = len(pads) // 2
        if ndim == 1:
            pad_begin = pads[0]
            pad_end = pads[1]
        elif ndim == 2:
            if pads[0] != 0 or pads[2] != 0:
                raise NotImplementedError(
                    "Pad: padding along the batch dimension (dim 0) is not supported."
                )
            pad_begin = pads[1]
            pad_end = pads[3]
        else:
            raise NotImplementedError(
                f"Pad: rank-{ndim} padding is not supported; "
                "only rank-1 and rank-2 (batch × features) are handled."
            )

        # ── Read constant fill value (opset >= 11 input[2], default 0) ────
        fill_val: float = 0.0
        if len(self.inputs) > 2 and self.inputs[2]:
            cv = self._variables.get_initializer_value(self.inputs[2])
            if cv is not None:
                fill_val = float(list(cv)[0]) if hasattr(cv, "__iter__") else float(cv)

        # ── Consume input ─────────────────────────────────────────────────
        data = self._variables.consume(self.inputs[0])
        if isinstance(data, VariablesGroup):
            in_cols = list(data.values())
        else:
            in_cols = [data]

        # ── Assemble output columns ────────────────────────────────────────
        fill_expr = ibis.literal(fill_val)
        out: dict[str, ibis.expr.types.Value] = {}

        for i in range(pad_begin):
            out[f"pad_begin_{i}"] = fill_expr
        for i, col in enumerate(in_cols):
            out[f"data_{i}"] = col
        for i in range(pad_end):
            out[f"pad_end_{i}"] = fill_expr

        self.set_output(ValueVariablesGroup(out))
