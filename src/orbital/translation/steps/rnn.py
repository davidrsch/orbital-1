"""Implementation of the ONNX RNN operator for fixed-length sequences.

Only the vanilla ("simple") RNN with a single update gate is supported:

    H_t = cell_act(X_t @ W.T + H_prev @ R.T + Wb + Rb)

where ``cell_act`` defaults to ``Tanh`` per the ONNX spec.

Weight layout (ONNX)
--------------------
* W : ``[1, H, I]``  — input weight matrix
* R : ``[1, H, H]``  — recurrent weight matrix
* B : ``[1, 2H]``    — ``[Wb (H,), Rb (H,)]`` (optional)

Limitations (raises :class:`NotImplementedError` otherwise)
------------------------------------------------------------
* Forward direction only (``direction="forward"``).
* Default activation only (``Tanh``).
* ``sequence_lens`` (input[4]) is not supported.
* Non-zero ``initial_h`` (input[5]) is not supported.

References
----------
https://onnx.ai/onnx/operators/onnx__RNN.html
"""

import ibis

from ..translator import Translator
from ..variables import VariablesGroup
from ._rnn_base import _get_flat_weights, _write_bidir_sequence_outputs, _write_sequence_outputs
from .tanh import _tanh


class RNNTranslator(Translator):
    """Translate the ONNX RNN operator by unrolling the recurrence."""

    def process(self) -> None:
        """Translate the RNN node, writing result(s) to the graph."""
        # https://onnx.ai/onnx/operators/onnx__RNN.html

        direction = str(self._attributes.get("direction", "forward"))
        if direction not in ("forward", "bidirectional"):
            raise NotImplementedError(
                f"RNN: direction={direction!r} is not supported; "
                "must be 'forward' or 'bidirectional'."
            )

        activations = self._attributes.get("activations", None)
        if activations and list(activations) not in (
            ["Tanh"],
            ["tanh"],
        ):
            raise NotImplementedError(
                "RNN: only the default activation [Tanh] is supported."
            )

        hidden_size = int(self._attributes["hidden_size"])
        H = hidden_size

        # ── Extract W [1, H, I] ───────────────────────────────────────────
        w_flat, w_dims = _get_flat_weights(self._variables, self.inputs[1], "RNN")
        num_dir_w, H_check, I = w_dims
        num_dir_expected = 2 if direction == "bidirectional" else 1
        if num_dir_w != num_dir_expected:
            raise ValueError(
                f"RNN: W has {num_dir_w} direction(s) but direction={direction!r} "
                f"requires exactly {num_dir_expected} direction(s)."
            )
        if H_check != H:
            raise ValueError(f"RNN: W dim[1]={H_check} expected hidden_size={H}.")

        # ── Extract R [1, H, H] ───────────────────────────────────────────
        r_flat, _ = _get_flat_weights(self._variables, self.inputs[2], "RNN")

        # ── Extract B [1, 2H] (optional) ─────────────────────────────────
        has_bias = len(self.inputs) > 3 and bool(self.inputs[3])
        b_flat: list = []
        if has_bias:
            b_val = self._variables.get_initializer_value(self.inputs[3])
            if b_val is not None and isinstance(b_val, (list, tuple)):
                b_flat = list(b_val)

        # ── Sequence length and initial-state guards ──────────────────────
        # ONNX RNN inputs: [X, W, R, B, sequence_lens, initial_h]
        if len(self.inputs) > 4 and bool(self.inputs[4]):
            raise NotImplementedError(
                "RNN: sequence_lens (input[4]) is not supported; "
                "all sequences must have the same fixed length T."
            )
        if len(self.inputs) > 5 and bool(self.inputs[5]):
            raise NotImplementedError(
                "RNN: non-zero initial_h (input[5]) is not supported; "
                "the initial hidden state is assumed to be all-zeros."
            )

        # ── Consume X input ───────────────────────────────────────────────
        x_val = self._variables.consume(self.inputs[0])
        if isinstance(x_val, VariablesGroup):
            x_exprs = list(x_val.values())
        else:
            x_exprs = [x_val]

        total_in = len(x_exprs)
        if I <= 0 or total_in % I != 0:
            raise ValueError(
                f"RNN: input group has {total_in} elements which is not divisible"
                f" by input_size={I}; cannot infer sequence length T."
            )
        T = total_in // I

        # W [num_directions, H, I] — offset by H*I per direction
        W_DIR = H * I
        # R [num_directions, H, H] — offset by H*H per direction
        R_DIR = H * H
        # B [num_directions, 2H] — offset by 2*H per direction
        B_DIR = 2 * H

        def _run_direction(
            d: int,
            x_seq: list,
        ) -> tuple[list, list]:
            """Unroll one RNN direction; returns (all_H, H_state)."""
            w_off = d * W_DIR
            r_off = d * R_DIR
            b_off = d * B_DIR

            def w_val(h: int, i: int) -> float:
                return float(w_flat[w_off + h * I + i])

            def r_val(h: int, hid: int) -> float:
                return float(r_flat[r_off + h * H + hid])

            def bias_in(h: int) -> float:
                return float(b_flat[b_off + h]) if b_flat else 0.0

            def bias_rec(h: int) -> float:
                return float(b_flat[b_off + H + h]) if b_flat else 0.0

            H_st: list[ibis.expr.types.NumericValue] = [ibis.literal(0.0) for _ in range(H)]
            all_H_dir: list[list[ibis.expr.types.NumericValue]] = []

            for t in range(len(x_seq) // I):
                x_t = x_seq[t * I : (t + 1) * I]

                pre_h = [
                    self._optimizer.fold_operation(
                        sum(
                            [x_t[i] * w_val(h, i) for i in range(I)]
                            + [H_st[hid] * r_val(h, hid) for hid in range(H)]
                        )
                        + bias_in(h)
                        + bias_rec(h)
                    )
                    for h in range(H)
                ]

                new_H = [_tanh(v) for v in pre_h]
                H_st = new_H
                all_H_dir.append(new_H)

            return all_H_dir, H_st

        # ── Run direction(s) ──────────────────────────────────────────────
        outputs = self.outputs  # may have 1 or 2 entries; some may be ""

        all_H_fwd, H_state_fwd = _run_direction(0, x_exprs)

        if direction == "bidirectional":
            x_bwd: list = []
            for t in reversed(range(T)):
                x_bwd.extend(x_exprs[t * I : (t + 1) * I])
            all_H_bwd_rev, H_state_bwd = _run_direction(1, x_bwd)
            all_H_bwd = list(reversed(all_H_bwd_rev))
            _write_bidir_sequence_outputs(
                self._variables, outputs,
                all_H_fwd, all_H_bwd, H_state_fwd, H_state_bwd,
            )
        else:
            _write_sequence_outputs(self._variables, outputs, all_H_fwd, H_state_fwd)
