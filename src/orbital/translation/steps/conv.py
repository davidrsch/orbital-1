"""Implementation of the Conv(1D) operator for fixed-length sequences."""

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup


class ConvTranslator(Translator):
    """Translate the ONNX Conv operator for 1-D convolutions.

    Supports only:
    - ``group=1``
    - ``dilations=[d]`` (single dilation value)
    - ``strides=[s]``  (single stride value)
    - ``pads=[p_left, p_right]`` (zero-padding on the width axis)

    The input ``X`` is expected to arrive as a :class:`~orbital.translation.variables.VariablesGroup`
    with ``C_in * W_in`` columns ordered ``(channel, position)`` so that
    ``X[c, w]`` maps to flat index ``c * W_in + w``.

    Raises :class:`NotImplementedError` for 2-D / 3-D convolutions (their
    presence is detected via W rank).
    """

    def process(self) -> None:
        """Translate the Conv node, writing the result to the graph."""
        # https://onnx.ai/onnx/operators/onnx__Conv.html
        group = int(self._attributes.get("group", 1))
        dilations = self._attributes.get("dilations", [1])
        pads = self._attributes.get("pads", [0, 0])
        strides = self._attributes.get("strides", [1])

        if group != 1:
            raise NotImplementedError("Conv: only group=1 is supported.")

        # ── Extract weights ────────────────────────────────────────────────
        w_tensor = self._variables.get_initializer(self.inputs[1])
        if w_tensor is None:
            raise ValueError("Conv: weight tensor W not found in initializers.")

        w_dims = list(w_tensor.dims)
        if len(w_dims) != 3:
            raise NotImplementedError(
                f"Conv: only 1-D convolutions (3-D weight tensor) are supported;"
                f" got weight shape {w_dims}."
            )
        c_out, c_in, k_w = w_dims

        w_flat = self._variables.get_initializer_value(self.inputs[1])
        if w_flat is None or not isinstance(w_flat, (list, tuple)):
            raise ValueError("Conv: weight values could not be read.")

        has_bias = len(self.inputs) > 2 and self.inputs[2] != ""
        bias_flat: list = []
        if has_bias:
            b_val = self._variables.get_initializer_value(self.inputs[2])
            if b_val is not None and isinstance(b_val, (list, tuple)):
                bias_flat = list(b_val)

        dil = int(dilations[0]) if dilations else 1
        stride = int(strides[0]) if strides else 1
        pad_l = int(pads[0]) if pads else 0
        pad_r = int(pads[1]) if len(pads) > 1 else 0

        # ── Consume input ─────────────────────────────────────────────────
        input_val = self._variables.consume(self.inputs[0])

        if isinstance(input_val, VariablesGroup):
            input_exprs = list(input_val.values())
        else:
            input_exprs = [input_val]

        total_in = len(input_exprs)
        if total_in % c_in != 0:
            raise ValueError(
                f"Conv: input group has {total_in} elements which is not divisible"
                f" by C_in={c_in}."
            )
        w_in = total_in // c_in
        # X[c, w] → flat index c * w_in + w
        inp = lambda c, w: input_exprs[c * w_in + w]  # noqa: E731

        # effective kernel width considering dilation
        k_eff = dil * (k_w - 1) + 1
        w_out = (w_in + pad_l + pad_r - k_eff) // stride + 1
        if w_out <= 0:
            raise ValueError(
                f"Conv: computed output width {w_out} is non-positive for"
                f" W_in={w_in}, kernel={k_w}, dilation={dil}, stride={stride},"
                f" pads={pads}."
            )

        # W[f, c, k] → flat index (f * c_in + c) * k_w + k
        def w_val(f: int, c: int, k: int) -> float:
            return float(w_flat[(f * c_in + c) * k_w + k])

        # ── Build output expressions ───────────────────────────────────────
        results: dict[str, ibis.expr.types.Value] = {}
        for f in range(c_out):
            b = float(bias_flat[f]) if bias_flat else 0.0
            for p in range(w_out):
                terms = []
                for c in range(c_in):
                    for k in range(k_w):
                        w_pos = p * stride + k * dil - pad_l
                        if 0 <= w_pos < w_in:
                            terms.append(inp(c, w_pos) * w_val(f, c, k))
                if terms:
                    dot = self._optimizer.fold_operation(sum(terms))
                else:
                    dot = ibis.literal(0.0)
                results[f"out_{f}_{p}"] = self._optimizer.fold_operation(dot + b)

        # Output order: (filter, position) ─ flat index f * w_out + p
        self.set_output(ValueVariablesGroup(results))
