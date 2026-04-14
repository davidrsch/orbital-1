"""Implementation of the ConvTranspose (1-D transposed convolution) operator.

For the SQL/tabular context, ConvTranspose is supported when:

- Weight rank is 3 (1-D transposed convolution only).
- ``group=1``.
- ``dilations=[d]`` (single dilation value).
- ``strides=[s]`` and ``pads=[p_left, p_right]`` with ``auto_pad`` in
  NOTSET / VALID / SAME_UPPER / SAME_LOWER.

Weight layout is ``(C_in, C_out, kW)`` — the transpose of Conv's
``(C_out, C_in, kW)`` layout.

The input ``X`` is expected as a :class:`~orbital.translation.variables.VariablesGroup`
with ``C_in * W_in`` columns in channel-major order (``X[c, w]`` → flat index
``c * W_in + w``).  The output is a :class:`~orbital.translation.variables.ValueVariablesGroup`
with ``C_out * W_out`` columns in the same channel-major order.

References
----------
https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
"""

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup


class ConvTransposeTranslator(Translator):
    """Translate the ONNX ConvTranspose operator for 1-D transposed convolutions."""

    def process(self) -> None:
        """Translate the ConvTranspose node, writing the result to the graph."""
        group: int = int(self._attributes.get("group", 1))
        auto_pad: str = str(self._attributes.get("auto_pad", "NOTSET"))
        dilations = self._attributes.get("dilations", [1])
        pads = self._attributes.get("pads", [0, 0])
        strides = self._attributes.get("strides", [1])
        output_padding = self._attributes.get("output_padding", [0])

        if group != 1:
            raise NotImplementedError("ConvTranspose: only group=1 is supported.")

        # ── Extract weights ──────────────────────────────────────────────────
        w_tensor = self._variables.get_initializer(self.inputs[1])
        if w_tensor is None:
            raise ValueError(
                "ConvTranspose: weight tensor W not found in initializers."
            )
        w_dims = list(w_tensor.dims)
        if len(w_dims) != 3:
            raise NotImplementedError(
                f"ConvTranspose: only 1-D (3-D weight tensor) is supported;"
                f" got weight shape {w_dims}."
            )
        # ConvTranspose weight layout: (C_in, C_out, kW)
        c_in, c_out, k_w = w_dims

        w_flat = self._variables.get_initializer_value(self.inputs[1])
        if w_flat is None or not isinstance(w_flat, (list, tuple)):
            raise ValueError("ConvTranspose: weight values could not be read.")

        has_bias = len(self.inputs) > 2 and self.inputs[2] != ""
        bias_flat: list = []
        if has_bias:
            b_val = self._variables.get_initializer_value(self.inputs[2])
            if b_val is not None and isinstance(b_val, (list, tuple)):
                bias_flat = list(b_val)

        dil = int(dilations[0]) if dilations else 1
        stride = int(strides[0]) if strides else 1
        out_pad = int(output_padding[0]) if output_padding else 0

        # ── Consume input ────────────────────────────────────────────────────
        input_val = self._variables.consume(self.inputs[0])
        if isinstance(input_val, VariablesGroup):
            input_exprs = list(input_val.values())
        else:
            input_exprs = [input_val]

        total_in = len(input_exprs)
        if total_in % c_in != 0:
            raise ValueError(
                f"ConvTranspose: input group has {total_in} elements which is not"
                f" divisible by C_in={c_in}."
            )
        w_in = total_in // c_in
        # X[c, w] → flat index c * w_in + w
        inp = lambda c, w: input_exprs[c * w_in + w]  # noqa: E731

        # effective kernel width (dil=1: k_eff = k_w)
        k_eff = dil * (k_w - 1) + 1

        # Compute explicit padding.
        if auto_pad == "VALID":
            pad_l, pad_r = 0, 0
        elif auto_pad == "NOTSET":
            pad_l = int(pads[0]) if pads else 0
            pad_r = int(pads[1]) if len(pads) > 1 else 0
        elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            # Infer from SAME semantics: input size determines output size.
            w_out_same = w_in * stride
            total_pad = max(0, k_eff - stride)
            if auto_pad == "SAME_UPPER":
                pad_l = total_pad // 2
                pad_r = total_pad - pad_l
            else:
                pad_r = total_pad // 2
                pad_l = total_pad - pad_r
        else:
            raise NotImplementedError(
                f"ConvTranspose: auto_pad='{auto_pad}' is not supported."
            )

        w_out = (w_in - 1) * stride - pad_l - pad_r + k_eff + out_pad
        if w_out <= 0:
            raise ValueError(
                f"ConvTranspose: computed output width {w_out} is non-positive for"
                f" W_in={w_in}, kernel={k_w}, dilation={dil}, stride={stride},"
                f" pads=[{pad_l},{pad_r}], output_padding={out_pad}."
            )

        # W[c_in, c_out, k] → flat index (c_in * c_out + c_out_idx) * k_w + k
        def w_val(ci: int, co: int, k: int) -> float:
            return float(w_flat[(ci * c_out + co) * k_w + k])

        # ── Build output expressions ─────────────────────────────────────────
        # For each output position o and output channel co:
        #   Y[co, o] = sum over ci, k: X[ci, i] * W[ci, co, k]
        #   where i = (o + pad_l - k * dil) / stride  (integer, in [0, w_in))
        results: dict[str, ibis.expr.types.Value] = {}
        for co in range(c_out):
            b = float(bias_flat[co]) if bias_flat else 0.0
            for o in range(w_out):
                terms = []
                for k in range(k_w):
                    # dil == 1, so: i * stride = o + pad_l - k
                    numerator = o + pad_l - k * dil
                    if numerator < 0:
                        continue
                    if numerator % stride != 0:
                        continue
                    i = numerator // stride
                    if not (0 <= i < w_in):
                        continue
                    for ci in range(c_in):
                        terms.append(inp(ci, i) * w_val(ci, co, k))
                if terms:
                    dot = self._optimizer.fold_operation(sum(terms))
                else:
                    dot = ibis.literal(0.0)
                results[f"out_{co}_{o}"] = self._optimizer.fold_operation(dot + b)

        # Output order: (output_channel, position) — flat index co * w_out + o
        self.set_output(ValueVariablesGroup(results))
