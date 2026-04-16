"""Implementation of the ONNX Tile operator for repeating feature sequences."""

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup


class TileTranslator(Translator):
    """Translate the ONNX Tile operator for repeating feature sequences.

    In the tabular 1-D context the input is a :class:`VariablesGroup` whose
    columns represent a flat ``[N]`` (rank-1) or ``[C × W]`` (rank-2) feature
    vector.  ``Tile`` repeats the tensor along each axis by the corresponding
    entry in the ``repeats`` constant tensor.

    Supported configurations
    ------------------------
    - ``repeats`` must be a constant initializer (integer 1-D tensor).
    - Rank-1 input: ``repeats = [k]`` — the whole column sequence is repeated
      ``k`` times.
    - Rank-2 input ``[batch, features]``: ``repeats = [1, k]`` — batch axis
      must be 1 (no batch repetition); features axis is repeated ``k`` times.
    - Higher rank or non-1 batch repetition raise :class:`NotImplementedError`.

    References
    ----------
    https://onnx.ai/onnx/operators/onnx__Tile.html
    """

    def process(self) -> None:
        """Translate the Tile node, writing the tiled group to the graph."""
        # ── Read repeats (always the second input, a constant) ─────────────
        repeats_raw = self._variables.get_initializer_value(self.inputs[1])
        if repeats_raw is None:
            raise NotImplementedError("Tile: 'repeats' must be a constant initializer.")
        repeats = [int(r) for r in repeats_raw]

        ndim = len(repeats)
        if ndim == 1:
            k = repeats[0]
        elif ndim == 2:
            if repeats[0] != 1:
                raise NotImplementedError(
                    "Tile: repeating along the batch dimension (dim 0) is not supported."
                )
            k = repeats[1]
        else:
            raise NotImplementedError(
                f"Tile: rank-{ndim} repeats not supported; "
                "only rank-1 and rank-2 [1, k] are handled."
            )

        # ── Consume input ──────────────────────────────────────────────────
        data = self._variables.consume(self.inputs[0])
        if isinstance(data, VariablesGroup):
            in_cols = list(data.values())
        else:
            in_cols = [data]

        # ── Build output: input sequence repeated k times ──────────────────
        out: dict[str, object] = {}
        for rep in range(k):
            for i, col in enumerate(in_cols):
                out[f"tile_{rep}_{i}"] = col

        self.set_output(ValueVariablesGroup(out))
