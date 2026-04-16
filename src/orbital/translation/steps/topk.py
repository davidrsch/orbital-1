"""Defines the translation step for the TopK operation.

For the tabular SQL context only axis=1 (column axis) is supported.
The approach finds the k-th largest (or smallest) column value per row
using CASE-WHEN comparison counting — no SQL window functions required.

References
----------
https://onnx.ai/onnx/operators/onnx__TopK.html
"""

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup

_MAX_K = 20  # practical cap to keep CASE-WHEN expressions tractable


class TopKTranslator(Translator):
    """Translate the ONNX TopK operator into column-wise ranked expressions.

    Limitations:
    - Only ``axis=1`` (the column / feature axis) is supported.
    - ``K`` must be a positive integer ≤ 20.
    - Input must be a dict (VariablesGroup) of columns.

    Outputs:
    - ``Values`` (output index 0): dict of the top-K column values per row.
    - ``Indices`` (output index 1): dict of 0-based column indices for each value.
    """

    def process(self) -> None:
        """Perform the translation and set Values and Indices outputs."""
        # https://onnx.ai/onnx/operators/onnx__TopK.html
        data = self._variables.consume(self.inputs[0])

        if not isinstance(data, dict):
            raise NotImplementedError(
                "TopKTranslator can only be applied to a group of columns"
            )

        # K is the second input (a scalar constant).
        k_raw = self._variables.consume(self.inputs[1])
        if isinstance(k_raw, ibis.Expr):
            try:
                k = int(k_raw.op().value)
            except Exception:
                k = int(k_raw.execute())
        else:
            k = int(k_raw)

        axis = int(self._attributes.get("axis", 1))
        largest = int(self._attributes.get("largest", 1))

        if axis != 1:
            raise NotImplementedError("TopKTranslator only supports axis=1")
        if k <= 0 or k > _MAX_K:
            raise NotImplementedError(
                f"TopKTranslator requires 1 <= K <= {_MAX_K}, got K={k}"
            )

        keys = list(data.keys())
        n = len(keys)
        k = min(k, n)

        values_dict: dict = {}
        indices_dict: dict = {}

        for rank in range(k):
            # For rank r (0-based), find the column that has exactly `rank` other
            # columns strictly better than it (ties broken by lower column index).
            # "Better" means strictly greater for largest=1, strictly less for
            # largest=0.
            val_cases: list = []
            idx_cases: list = []
            for col_idx, col_key in enumerate(keys):
                # Count how many columns are strictly better than this one.
                better_count_expr = None
                for other_idx, other_key in enumerate(keys):
                    if other_idx == col_idx:
                        continue
                    if largest:
                        strictly_better = data[other_key] > data[col_key]
                    else:
                        strictly_better = data[other_key] < data[col_key]
                    # Tie-breaking: a column with a lower index is considered
                    # "better" when values are equal.
                    tie_break = (data[other_key] == data[col_key]) & ibis.literal(
                        other_idx < col_idx
                    )
                    counts_as_better = strictly_better | tie_break
                    better_count_expr = (
                        counts_as_better.cast("int64")
                        if better_count_expr is None
                        else better_count_expr + counts_as_better.cast("int64")
                    )

                if better_count_expr is None:
                    # Only one column — it always wins rank 0.
                    cond = ibis.literal(True)
                else:
                    cond = better_count_expr == ibis.literal(rank)

                val_cases.append((cond, data[col_key]))
                idx_cases.append((cond, ibis.literal(col_idx)))

            # Build CASE expressions for this rank.
            # ibis.cases expects (condition, result) pairs; the last result is
            # the else clause, so we use the final pair's result as else.
            *leading_val, (last_val_cond, last_val_result) = val_cases
            *leading_idx, (last_idx_cond, last_idx_result) = idx_cases

            if not leading_val:
                val_expr = last_val_result
                idx_expr = last_idx_result
            else:
                val_expr = ibis.cases(*leading_val, else_=last_val_result)
                idx_expr = ibis.cases(*leading_idx, else_=last_idx_result)

            values_dict[rank] = val_expr
            indices_dict[rank] = idx_expr

        # Write outputs.  outputs[0] = Values, outputs[1] = Indices.
        self._variables[self.outputs[0]] = ValueVariablesGroup(values_dict)
        if len(self.outputs) > 1 and self.outputs[1]:
            self._variables[self.outputs[1]] = ValueVariablesGroup(indices_dict)
