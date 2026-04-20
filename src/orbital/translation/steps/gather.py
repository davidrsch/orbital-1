"""Defines the translation step for the Gather operation."""

import ibis

from orbital.translation.variables import ValueVariablesGroup, VariablesGroup

from ..translator import Translator


class GatherTranslator(Translator):
    """Processes a Gather node and updates the variables with the output expression.

    Supports two axes:

    axis=1 (column selection):
        The first operand is a column group or single column; the second is a
        constant integer index.  Returns the column at that index.

    axis=0 (row / embedding lookup):
        The first operand must be a 2-D constant initializer ``[vocab, embed]``.
        The second operand can be:
        - a constant integer index → returns the chosen row as literal columns.
        - a variable integer column → returns per-row CASE expressions (embedding).
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Gather.html

        axis = self._attributes.get("axis", 0)

        if axis == 1:
            # ── axis=1: pick a single column by constant index ────────────
            expr = self._variables.consume(self.inputs[0])
            idx = self._variables.get_initializer_value(self.inputs[1])
            if not isinstance(idx, (tuple, list)) or len(idx) != 1:
                raise NotImplementedError(
                    "Gather second operand must a list of one element"
                )

            # Limitation: only single-column gather on axis=1 is supported.
            # Multi-index gather would need to project several columns into a
            # VariablesGroup result, but SQL has no native "select by positional
            # index list" shape — each target column would need its own CASE
            # chain keyed on the source column's position, and the upstream
            # pipeline would then have to track the resulting column-group
            # schema.  The caller above already rejects any index list with
            # length != 1, so reaching this line guarantees a single index.
            idx = idx[0]
            if not isinstance(idx, int):
                raise ValueError("Gather: index must be an integer constant")

            if isinstance(expr, VariablesGroup):
                keys = list(expr.keys())
                if idx < 0 or idx >= len(keys):
                    raise IndexError("Gather: index out of bounds")
                self.set_output(expr[keys[idx]])
            else:
                if idx != 0:
                    raise NotImplementedError(
                        f"Gather: index {idx} not supported for single columns"
                    )
                self.set_output(expr)

        elif axis == 0:
            # ── axis=0: row / embedding lookup from a 2-D constant matrix ─
            data_tensor = self._variables.get_initializer(self.inputs[0])
            data_flat = self._variables.get_initializer_value(self.inputs[0])

            if data_tensor is None or not isinstance(data_flat, (list, tuple)):
                raise NotImplementedError(
                    "Gather (axis=0): data must be a constant initializer "
                    "(2-D embedding weight matrix)."
                )

            dims = list(data_tensor.dims)
            if len(dims) != 2:
                raise NotImplementedError(
                    f"Gather (axis=0): only 2-D data tensors are supported, "
                    f"got shape {dims}."
                )
            vocab_size, embed_dim = dims

            idx_raw = self._variables.consume(self.inputs[1])

            if isinstance(idx_raw, (list, tuple)):
                # Constant index: return the specified row as literal columns.
                if len(idx_raw) != 1 or not isinstance(idx_raw[0], int):
                    raise NotImplementedError(
                        "Gather (axis=0): constant indices must be a single integer."
                    )
                row = idx_raw[0]
                if row < 0 or row >= vocab_size:
                    raise IndexError("Gather (axis=0): row index out of bounds.")
                self.set_output(
                    ValueVariablesGroup(
                        {
                            f"emb_{j}": ibis.literal(
                                float(data_flat[row * embed_dim + j])
                            )
                            for j in range(embed_dim)
                        }
                    )
                )
            else:
                # Variable index: build per-column CASE/WHEN expressions.
                idx_col = idx_raw  # ibis NumericValue (integer column)
                results = {}
                for j in range(embed_dim):
                    cases = [
                        (
                            idx_col == ibis.literal(row),
                            ibis.literal(float(data_flat[row * embed_dim + j])),
                        )
                        for row in range(vocab_size)
                    ]
                    results[f"emb_{j}"] = ibis.cases(
                        *cases, else_=ibis.null().cast("float64")
                    )
                self.set_output(ValueVariablesGroup(results))

        else:
            raise NotImplementedError(
                f"Gather: axis={axis} is not supported "
                f"(supported: axis=0 for embedding lookup, axis=1 for column selection)."
            )
