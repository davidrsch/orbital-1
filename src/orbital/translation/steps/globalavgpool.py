"""Implementation of the GlobalAveragePool operator.

In the SQL/tabular context, each input row represents one sample.  A
``VariablesGroup`` of C columns corresponds to C channels (or sequence
positions), and GlobalAveragePool reduces them to their per-row mean —
collapsing the spatial/sequence dimension to a single scalar.

For a single-column input (no group), the operation is the identity.

References
----------
https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html
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


class GlobalAveragePoolTranslator(Translator):
    """Translate the ONNX GlobalAveragePool operator: per-row mean across all channels."""

    def process(self) -> None:
        """Translate the GlobalAveragePool node, writing result to the graph."""
        data = self._variables.consume(self.inputs[0])

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            keys = list(data.keys())
            cols = list(data.values())
            n = len(cols)
            if n == 0:
                raise ValueError(
                    "GlobalAveragePool: input group must have at least one column."
                )

            # Detect multi-channel layout: keys from Conv/ConvTranspose follow
            # the ``out_{channel}_{position}`` pattern.  When multiple channels
            # are present, average over the spatial positions *per channel* and
            # emit one output expression per channel.
            channel_groups = _group_by_channel(keys)
            if channel_groups is not None and len(channel_groups) > 1:
                results: dict[str, ibis.expr.types.Value] = {}
                for c in sorted(channel_groups.keys()):
                    group_cols = [data[k] for k in channel_groups[c]]
                    mean_expr: ibis.expr.types.NumericValue = group_cols[0]
                    for col in group_cols[1:]:
                        mean_expr = mean_expr + col
                    mean_expr = mean_expr / ibis.literal(float(len(group_cols)))
                    results[f"out_{c}_0"] = self._optimizer.fold_operation(mean_expr)
                self.set_output(ValueVariablesGroup(results))
            else:
                # Single-channel or flat features: average all columns to one scalar.
                mean_expr = cols[0]
                for col in cols[1:]:
                    mean_expr = mean_expr + col
                mean_expr = mean_expr / ibis.literal(float(n))
                self.set_output(self._optimizer.fold_operation(mean_expr))
        else:
            # Single column: already reduced.
            self.set_output(data)
