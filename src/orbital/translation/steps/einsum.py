"""Implementation of the ONNX Einsum operator.

Supports the subset of Einstein summation equations that arise in
row-by-row tabular scoring:

  ``ab,bc->ac``   — standard matrix multiply (equivalent to Gemm/MatMul)
  ``...b,bc->...c`` — batch-style matmul emitted by some frameworks
  ``abc,cd->abd`` — (batch, seq, d_k) × (d_k, d_v) matmul used in attention

In all three cases the *second* input must be a constant weight tensor.
Dynamic second inputs (e.g. attention key/value matrices that are row-data)
raise ``NotImplementedError``.
"""

from __future__ import annotations

import re

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup

# ---------------------------------------------------------------------------
# Supported equation patterns
# ---------------------------------------------------------------------------

# Groups: (lhs_free, contract_dim, rhs_free)
#   "ab,bc->ac"      → lhs_free="a", contract="b", rhs_free="c"
#   "abc,cd->abd"    → lhs_free="ab", contract="c", rhs_free="d"
_RE_STANDARD = re.compile(r"^([a-z.]+)([a-z]),\2([a-z]+)->([a-z.]+\3)$")


def _parse_equation(eq: str) -> tuple[str, str]:
    """Return ``(batch_prefix, rhs_out)`` for supported equations.

    Raises ``NotImplementedError`` for unsupported equations.
    """
    # Normalise whitespace
    eq = eq.replace(" ", "")
    # Handle ellipsis prefix: "...b,bc->...c"
    eq_norm = eq.replace("...", "_batch_")

    # Try to parse lhs1,lhs2->rhs
    m = re.match(r"^([a-z_]+),([a-z_]+)->([a-z_]+)$", eq_norm)
    if m is None:
        raise NotImplementedError(
            f"EinsumTranslator: cannot parse equation '{eq}'. "
            "Only 2-operand equations of the form 'ab,bc->ac' are supported."
        )
    lhs1, lhs2, rhs = m.groups()
    # Find the contracted index (last index of lhs1 that is also first of lhs2)
    # For our supported cases lhs1[-1] == lhs2[0] is the contraction dimension.
    if lhs1[-1] != lhs2[0]:
        raise NotImplementedError(
            f"EinsumTranslator: unsupported contraction in equation '{eq}'. "
            "Expected the last index of the left operand to match the first index of "
            "the right operand (e.g. 'ab,bc->ac' or 'abc,cd->abd')."
        )
    contract_idx = lhs1[-1]
    free_left = lhs1[:-1]  # e.g. "a", "ab", "_batch_a"
    free_right = lhs2[1:]  # e.g. "c", "d"
    expected_rhs = free_left + free_right
    if rhs != expected_rhs:
        raise NotImplementedError(
            f"EinsumTranslator: rhs '{rhs}' does not match expected '{expected_rhs}' "
            f"for equation '{eq}'."
        )
    _ = contract_idx  # consumed implicitly — only the shape matters
    return free_left, free_right


class EinsumTranslator(Translator):
    """Translate the ONNX Einsum operator for supported 2-operand equations.

    The second input must be a constant weight matrix stored as an
    ONNX initializer.  The result is a :class:`ValueVariablesGroup`
    whose keys are ``out_0``, ``out_1``, … (or a bare column expression
    when the output has a single element).
    """

    def process(self) -> None:
        """Translate Einsum and write result to the graph variables."""
        # https://onnx.ai/onnx/operators/onnx__Einsum.html
        equation: str = self._attributes.get("equation", "")
        if not equation:
            raise ValueError("EinsumTranslator: missing 'equation' attribute.")

        _parse_equation(equation)  # validate early; raises on unsupported forms

        if len(self.inputs) != 2:
            raise NotImplementedError(
                "EinsumTranslator only supports 2-operand equations."
            )

        weight_name = self.inputs[1]

        # The second operand must be a constant weight tensor.
        if self._variables.peek_variable(weight_name) is not None:
            raise NotImplementedError(
                "EinsumTranslator: the second operand must be a constant weight "
                "initializer.  Dynamic second operands are not supported."
            )

        w_tensor = self._variables.get_initializer(weight_name)
        if w_tensor is None:
            raise ValueError(
                f"EinsumTranslator: weight '{weight_name}' not found in initializers."
            )
        w_shape = list(w_tensor.dims)
        if len(w_shape) != 2:
            raise ValueError(
                f"EinsumTranslator: weight must be 2-D, got shape {w_shape}. "
                "Higher-rank weight tensors are not supported."
            )

        w_flat = self._variables.get_initializer_value(weight_name)
        if w_flat is None or not isinstance(w_flat, (list, tuple)):
            raise ValueError(
                "EinsumTranslator: weight tensor values could not be read."
            )

        input_dim, output_dim = w_shape  # shape is (K, N) → rows=K, cols=N

        # Consume the left operand (row data).
        left_operand = self._variables.consume(self.inputs[0])

        if isinstance(left_operand, dict):
            left_exprs: list[ibis.expr.types.NumericValue] = list(left_operand.values())
        else:
            left_exprs = [left_operand]  # type: ignore[list-item]

        if len(left_exprs) != input_dim:
            raise ValueError(
                f"EinsumTranslator: left operand has {len(left_exprs)} elements "
                f"but weight expects {input_dim} inputs."
            )

        # Compute output_j = sum_i( left[i] * W[i, j] ) for j in 0..output_dim-1
        results = []
        for j in range(output_dim):
            terms = [
                self._optimizer.fold_operation(
                    left_exprs[i] * float(w_flat[i * output_dim + j])
                )
                for i in range(input_dim)
            ]
            results.append(self._optimizer.fold_operation(sum(terms)))

        if output_dim == 1:
            self.set_output(results[0])
        else:
            self.set_output(
                ValueVariablesGroup({f"out_{j}": results[j] for j in range(output_dim)})
            )
