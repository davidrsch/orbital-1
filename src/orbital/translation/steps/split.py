"""Implementation of the Split operator.

For the SQL/tabular context, only splitting along the feature axis
(axis=-1 or axis=1) is supported.  The input must be a
``VariablesGroup``; the operator partitions it into N sub-groups
according to the ``split`` attribute (or ``num_outputs`` when the
attribute is absent).

References
----------
https://onnx.ai/onnx/operators/onnx__Split.html
"""

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup


class SplitTranslator(Translator):
    """Translate the ONNX Split operator: partition a VariablesGroup."""

    def process(self) -> None:
        """Translate the Split node, writing sub-group outputs to the graph."""
        data = self._variables.consume(self.inputs[0])

        axis = int(self._attributes.get("axis", 0))
        if axis not in (-1, 1):
            raise NotImplementedError(
                f"Split: only axis=-1 or axis=1 (feature axis) is supported, got {axis}"
            )

        if not isinstance(data, VariablesGroup):
            raise NotImplementedError(
                "Split: input must be a VariablesGroup (multi-column tensor)."
            )

        cols = list(data.values())
        keys = list(data.keys())
        N = len(cols)

        # split sizes from attribute (opset < 18) or second input (opset >= 18).
        split_sizes = self._attributes.get("split", None)
        if split_sizes is not None:
            split_sizes = list(split_sizes)

        if len(self.inputs) > 1 and bool(self.inputs[1]):
            split_val = self._variables.get_initializer_value(self.inputs[1])
            if split_val is not None:
                split_sizes = [int(s) for s in split_val]

        num_outputs = len(self.outputs)

        if split_sizes is None:
            # Equal split across num_outputs.
            if num_outputs == 0:
                return
            if N % num_outputs != 0:
                raise ValueError(
                    f"Split: {N} columns cannot be split equally into {num_outputs} outputs."
                )
            chunk = N // num_outputs
            split_sizes = [chunk] * num_outputs

        if sum(split_sizes) != N:
            raise ValueError(
                f"Split: split sizes {split_sizes} sum to {sum(split_sizes)}"
                f" but input has {N} columns."
            )

        offset = 0
        for out_name, size in zip(self.outputs, split_sizes):
            if not out_name:
                offset += size
                continue
            chunk_cols = cols[offset : offset + size]
            chunk_keys = keys[offset : offset + size]
            self._variables[out_name] = ValueVariablesGroup(
                {k: v for k, v in zip(chunk_keys, chunk_cols)}
            )
            offset += size
