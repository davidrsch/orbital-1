"""Implementation of the AveragePool operator.

For the SQL/tabular context, AveragePool is supported when:

- ``kernel_shape`` has all dimensions equal to 1 (pass-through), or
- The input is a ``VariablesGroup`` of *L* columns and a 1-D
  ``kernel_shape = [k]``; covers both global average (k == L) and sliding-window
  (k < L) cases.  The input is treated as a single-channel sequence of length L.

Other shapes (2-D, 3-D, or dilated) raise ``NotImplementedError``.

References
----------
https://onnx.ai/onnx/operators/onnx__AveragePool.html
"""

import math

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


class AveragePoolTranslator(Translator):
    """Translate the ONNX AveragePool operator for the SQL/tabular context."""

    def process(self) -> None:
        """Translate the AveragePool node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])
        kernel_shape: list[int] = list(self._attributes.get("kernel_shape", [1]))

        # Trivial case: kernel = 1 everywhere → identity.
        if all(k == 1 for k in kernel_shape):
            self.set_output(data)
            return

        if not isinstance(data, VariablesGroup):
            raise NotImplementedError(
                f"AveragePool: kernel_shape={kernel_shape} is not supported for a "
                "single-column input."
            )

        if len(kernel_shape) != 1:
            raise NotImplementedError(
                f"AveragePool: only 1-D pooling (one kernel_shape value) is supported;"
                f" got kernel_shape={kernel_shape}."
            )

        data = NumericVariablesGroup(data)
        cols = list(data.values())
        n = len(cols)
        k = kernel_shape[0]

        dilations: list[int] = list(self._attributes.get("dilations", [1]))
        dil = int(dilations[0]) if dilations else 1

        auto_pad = str(self._attributes.get("auto_pad", "NOTSET"))
        pads: list[int] = list(self._attributes.get("pads", [0, 0]))
        strides: list[int] = list(self._attributes.get("strides", [k]))
        ceil_mode: int = int(self._attributes.get("ceil_mode", 0))
        count_include_pad: int = int(self._attributes.get("count_include_pad", 0))

        stride = int(strides[0]) if strides else k
        k_eff = dil * (k - 1) + 1

        if auto_pad == "VALID":
            pad_l, pad_r = 0, 0
        elif auto_pad == "NOTSET":
            pad_l = int(pads[0]) if pads else 0
            pad_r = int(pads[1]) if len(pads) > 1 else 0
        elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            l_out_nom = (n + stride - 1) // stride
            total_pad = max(0, (l_out_nom - 1) * stride + k_eff - n)
            if auto_pad == "SAME_UPPER":
                pad_l, pad_r = total_pad // 2, total_pad - total_pad // 2
            else:
                pad_r, pad_l = total_pad // 2, total_pad - total_pad // 2
        else:
            raise NotImplementedError(
                f"AveragePool: auto_pad='{auto_pad}' is not supported."
            )

        padded = n + pad_l + pad_r
        if ceil_mode:
            l_out = math.ceil((padded - k_eff) / stride) + 1
        else:
            l_out = (padded - k_eff) // stride + 1

        if l_out <= 0:
            raise ValueError(
                f"AveragePool: computed output length {l_out} is non-positive for "
                f"L_in={n}, kernel={k}, stride={stride}, pads=[{pad_l},{pad_r}]."
            )

        results: dict[str, ibis.expr.types.Value] = {}
        for p in range(l_out):
            valid: list[ibis.expr.types.NumericValue] = []
            for kk in range(k):
                pos = p * stride + kk * dil - pad_l
                if 0 <= pos < n:
                    valid.append(cols[pos])
            if not valid:
                results[f"out_0_{p}"] = ibis.literal(0.0)
                continue
            total: ibis.expr.types.NumericValue = valid[0]
            for v in valid[1:]:
                total = total + v
            denom = k if count_include_pad else len(valid)
            results[f"out_0_{p}"] = self._optimizer.fold_operation(
                total / ibis.literal(float(denom))
            )

        self.set_output(ValueVariablesGroup(results))
