"""Implementation of the GlobalMaxPool operator.

Reduces a ``VariablesGroup`` of C columns (channels / sequence positions) to a
single per-row maximum value — the SQL analogue of global max-pooling over the
spatial / sequence dimension.

For a single-column input the operation is the identity.

References
----------
https://onnx.ai/onnx/operators/onnx__GlobalMaxPool.html
"""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


def _group_by_channel(keys: list[str]) -> "dict[int, list[str]] | None":
    """Parse keys shaped ``out_{c}_{p}`` and return a channel→[keys] mapping.

    Returns ``None`` if any key does not match the expected pattern.
    """
    groups: dict[int, list[str]] = {}
    for key in keys:
        parts = key.split("_")
        if len(parts) != 3 or parts[0] != "out":
            return None
        try:
            c, p = int(parts[1]), int(parts[2])
        except ValueError:
            return None
        groups.setdefault(c, [])
        groups[c].append(key)
    return groups


class GlobalMaxPoolTranslator(Translator):
    """Translate the ONNX GlobalMaxPool operator: per-row maximum across all channels."""

    def process(self) -> None:
        """Translate the GlobalMaxPool node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            keys = list(data.keys())
            cols = list(data.values())
            if not cols:
                raise ValueError(
                    "GlobalMaxPool: input group must have at least one column."
                )

            # Detect multi-channel layout: keys from Conv/ConvTranspose follow
            # the ``out_{channel}_{position}`` pattern.  When multiple channels
            # are present, take the maximum over spatial positions *per channel*
            # and emit one output expression per channel.
            channel_groups = _group_by_channel(keys)
            if channel_groups is not None and len(channel_groups) > 1:
                results: dict[str, ibis.expr.types.Value] = {}
                for c in sorted(channel_groups.keys()):
                    group_cols = [data[k] for k in channel_groups[c]]
                    max_expr: ibis.expr.types.NumericValue = group_cols[0]
                    for col in group_cols[1:]:
                        max_expr = ibis.greatest(max_expr, col)
                    results[f"out_{c}_0"] = self._optimizer.fold_operation(max_expr)
                self.set_output(ValueVariablesGroup(results))
            else:
                # Single-channel or flat features: maximum across all columns.
                result: ibis.expr.types.NumericValue = cols[0]
                for col in cols[1:]:
                    result = ibis.greatest(result, col)
                self.set_output(self._optimizer.fold_operation(result))
        else:
            self.set_output(data)
