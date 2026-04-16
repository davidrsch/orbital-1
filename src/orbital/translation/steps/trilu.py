"""Implementation of the ONNX Trilu operator.

Trilu applies a triangular mask to a 2-D matrix, zeroing elements above
(``upper=1``, the default) or below (``upper=0``) the k-th diagonal.

In the orbital tabular scoring context Trilu is almost always applied to
a *constant* weight tensor (an ONNX initializer), so the output is a new
constant group of ibis literal expressions.  Dynamic (row-data) inputs
are also supported: each element in a :class:`ValueVariablesGroup` is
either kept as-is or replaced with ``ibis.literal(0.0)`` according to
its (row, col) position.
"""

from __future__ import annotations

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup


class TriluTranslator(Translator):
    """Translate the ONNX Trilu operator.

    The input tensor must be 2-D (a flat weight matrix stored as an
    initializer) or a :class:`ValueVariablesGroup` whose length equals
    ``rows * cols``.  The shape is inferred from the initializer dims or
    from the ``shape`` attribute when present.

    Elements that fall *outside* the triangular region are replaced with
    ``ibis.literal(0.0)``; elements *inside* are kept unchanged.
    """

    def process(self) -> None:
        """Translate Trilu and write result to the graph variables."""
        # https://onnx.ai/onnx/operators/onnx__Trilu.html
        upper: int = int(self._attributes.get("upper", 1))

        input_name = self.inputs[0]

        # k (diagonal offset) may be supplied as the second input (an initializer scalar).
        k = 0
        if len(self.inputs) >= 2 and self.inputs[1]:
            k_val = self._variables.get_initializer_value(self.inputs[1])
            if (
                k_val is not None
                and isinstance(k_val, (list, tuple))
                and len(k_val) == 1
            ):
                k = int(k_val[0])
            elif isinstance(k_val, (int, float)):
                k = int(k_val)

        # ---- Case 1: constant initializer input --------------------------------
        init_tensor = self._variables.get_initializer(input_name)
        if init_tensor is not None:
            shape = list(init_tensor.dims)
            if len(shape) != 2:
                raise ValueError(f"Trilu: initializer must be 2-D, got shape {shape}.")
            rows, cols = shape
            flat = self._variables.get_initializer_value(input_name)
            if flat is None or not isinstance(flat, (list, tuple)):
                raise ValueError("Trilu: could not read initializer values.")
            results: dict[str, ibis.Expr] = {}
            for r in range(rows):
                for c in range(cols):
                    idx = r * cols + c
                    val = float(flat[idx])
                    if _keep(r, c, k, upper):
                        results[f"out_{idx}"] = ibis.literal(val)
                    else:
                        results[f"out_{idx}"] = ibis.literal(0.0)
            self.set_output(ValueVariablesGroup(results))
            return

        # ---- Case 2: dynamic ValueVariablesGroup input -------------------------
        operand = self._variables.consume(input_name)
        if not isinstance(operand, ValueVariablesGroup):
            raise NotImplementedError(
                "Trilu: input must be an ONNX initializer or a ValueVariablesGroup. "
                "Scalar / bare-column inputs are not supported."
            )

        items = list(operand.items())
        n = len(items)
        # Determine shape: prefer square; users may pass 'rows' via options,
        # but we have no attribute for it.  Default: assume square.
        import math

        sqrt_n = math.isqrt(n)
        if sqrt_n * sqrt_n == n:
            rows = cols = sqrt_n
        else:
            raise NotImplementedError(
                f"Trilu: cannot infer 2-D shape from a flat group of {n} elements. "
                "Only square shapes are auto-detected for dynamic inputs."
            )

        result_items: dict[str, ibis.Expr] = {}
        for idx, (key, expr) in enumerate(items):
            r = idx // cols
            c = idx % cols
            if _keep(r, c, k, upper):
                result_items[key] = expr
            else:
                result_items[key] = ibis.literal(0.0)

        self.set_output(ValueVariablesGroup(result_items))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _keep(row: int, col: int, k: int, upper: int) -> bool:
    """Return True if element (row, col) is *inside* the triangular region."""
    # upper=1: keep where col >= row + k  (i.e. on or above the k-th diagonal)
    # upper=0: keep where col <= row + k  (i.e. on or below the k-th diagonal)
    if upper:
        return col >= row + k
    else:
        return col <= row + k
