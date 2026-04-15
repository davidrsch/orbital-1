"""Implementation of the LSTM operator for fixed-length sequences."""

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup
from ._rnn_base import (
    _get_flat_weights,
    _resolve_rnn_activation,
    _write_bidir_sequence_outputs,
    _write_sequence_outputs,
)


class LSTMTranslator(Translator):
    """Translate the ONNX LSTM operator by unrolling the recurrence.

    Limitations (raises :class:`NotImplementedError` otherwise):
    - Forward and reverse directions only (``direction="forward"`` or
      ``direction="reverse"`` or ``direction="bidirectional"``).
    - Default activations only (Sigmoid/Tanh/Tanh).
    - ``input_forget=0`` (default; independent input and forget gates).
    - No peephole connections (``P`` input must be absent / empty).
    - Sequence length ``T`` must be statically recoverable from the input
      group size (``len(X_group) // input_size``).

    ONNX gate order (IOFC):
    - i gate  → rows ``[0 : H]``
    - o gate  → rows ``[H : 2H]``
    - f gate  → rows ``[2H : 3H]``
    - c gate  → rows ``[3H : 4H]``

    ONNX ``B`` layout ``[1, 8H]``:
    - ``[0 : 4H]``  → input biases   (IOFC)
    - ``[4H : 8H]`` → recurrent biases (IOFC)
    """

    # Maps ONNX gate index (IOFC) to position within the four-gate blocks
    _GATE_I = 0
    _GATE_O = 1
    _GATE_F = 2
    _GATE_C = 3

    def process(self) -> None:
        """Translate the LSTM node, writing result(s) to the graph."""
        # https://onnx.ai/onnx/operators/onnx__LSTM.html

        direction = str(self._attributes.get("direction", "forward"))
        if direction not in ("forward", "reverse", "bidirectional"):
            raise NotImplementedError(
                f"LSTM: direction={direction!r} is not supported; "
                "must be 'forward', 'reverse', or 'bidirectional'."
            )

        activations = self._attributes.get("activations", None)
        act_names = list(activations) if activations else ["Sigmoid", "Tanh", "Tanh"]
        act_alphas = list(self._attributes.get("activation_alpha", []))
        act_betas = list(self._attributes.get("activation_beta", []))

        def _get_act(idx: int, default_name: str):
            n = act_names[idx] if idx < len(act_names) else default_name
            a = act_alphas[idx] if idx < len(act_alphas) else None
            b = act_betas[idx] if idx < len(act_betas) else None
            return _resolve_rnn_activation(n, a, b)

        # ONNX LSTM activation order: [f=gate(i/o/f), g=cell(c), h=out]
        # For bidirectional, activations [3,4,5] override the backward direction.
        _act_gate = _get_act(0, "Sigmoid")
        _act_cell = _get_act(1, "Tanh")
        _act_out = _get_act(2, "Tanh")
        _act_gate_bwd = _get_act(3, "Sigmoid") if len(act_names) > 3 else _act_gate
        _act_cell_bwd = _get_act(4, "Tanh") if len(act_names) > 4 else _act_cell
        _act_out_bwd = _get_act(5, "Tanh") if len(act_names) > 5 else _act_out

        input_forget = int(self._attributes.get("input_forget", 0))
        if input_forget != 0:
            raise NotImplementedError("LSTM: input_forget=1 (coupled gates) is not supported.")

        hidden_size = int(self._attributes["hidden_size"])
        H = hidden_size

        # ── Extract W [num_directions, 4H, I] ────────────────────────────────────
        w_flat, w_dims = _get_flat_weights(self._variables, self.inputs[1], "LSTM")
        num_dir_w, four_h, I = w_dims
        num_dir_expected = 2 if direction == "bidirectional" else 1
        if num_dir_w != num_dir_expected:
            raise ValueError(
                f"LSTM: W has {num_dir_w} direction(s) but direction={direction!r} "
                f"requires exactly {num_dir_expected} direction(s)."
            )
        if four_h != 4 * H:
            raise ValueError(f"LSTM: W dim[1]={four_h} expected 4*hidden_size={4*H}.")

        # ── Extract R [1, 4H, H] ──────────────────────────────────────────
        r_flat, _ = _get_flat_weights(self._variables, self.inputs[2], "LSTM")

        # ── Extract B [1, 8H] (optional) ─────────────────────────────────
        has_bias = len(self.inputs) > 3 and bool(self.inputs[3])
        b_flat: list = []
        if has_bias:
            b_val = self._variables.get_initializer_value(self.inputs[3])
            if b_val is not None and isinstance(b_val, (list, tuple)):
                b_flat = list(b_val)

        # ── Peephole connections not supported ────────────────────────────
        # ONNX LSTM inputs: [X, W, R, B, sequence_lens, initial_h, initial_c, P]
        # P (peephole) is at index 7, NOT index 4 (sequence_lens).
        if len(self.inputs) > 7 and bool(self.inputs[7]):
            raise NotImplementedError("LSTM: peephole connections (P input) are not supported.")

        # ── Sequence length and initial state guards ──────────────────────
        # ONNX LSTM inputs: [X, W, R, B, sequence_lens, initial_h, initial_c, P]
        if len(self.inputs) > 4 and bool(self.inputs[4]):
            raise NotImplementedError(
                "LSTM: sequence_lens (input[4]) is not supported; "
                "all sequences must have the same fixed length T."
            )
        if len(self.inputs) > 5 and bool(self.inputs[5]):
            raise NotImplementedError(
                "LSTM: non-zero initial_h (input[5]) is not supported; "
                "the initial hidden state is assumed to be all-zeros."
            )
        if len(self.inputs) > 6 and bool(self.inputs[6]):
            raise NotImplementedError(
                "LSTM: non-zero initial_c (input[6]) is not supported; "
                "the initial cell state is assumed to be all-zeros."
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
                f"LSTM: input group has {total_in} elements which is not divisible"
                f" by input_size={I}; cannot infer sequence length T."
            )
        T = total_in // I

        # W [num_directions, 4H, I] — offset by 4*H*I per direction
        W_DIR = 4 * H * I
        # R [num_directions, 4H, H] — offset by 4*H*H per direction
        R_DIR = 4 * H * H
        # B [num_directions, 8H] — offset by 8*H per direction
        B_DIR = 8 * H

        def _run_direction(
            d: int,
            x_seq: list,
            f_gate,
            g_cell,
            h_out,
        ) -> tuple[list, list, list]:
            """Unroll one LSTM direction; returns (all_H, H_state, C_state)."""
            w_off = d * W_DIR
            r_off = d * R_DIR
            b_off = d * B_DIR

            def w_val(g: int, h: int, i: int) -> float:
                return float(w_flat[w_off + g * H * I + h * I + i])

            def r_val(g: int, h: int, hid: int) -> float:
                return float(r_flat[r_off + g * H * H + h * H + hid])

            def bias_in(g: int, h: int) -> float:
                return float(b_flat[b_off + g * H + h]) if b_flat else 0.0

            def bias_rec(g: int, h: int) -> float:
                return float(b_flat[b_off + 4 * H + g * H + h]) if b_flat else 0.0

            H_st: list[ibis.expr.types.NumericValue] = [ibis.literal(0.0) for _ in range(H)]
            C_st: list[ibis.expr.types.NumericValue] = [ibis.literal(0.0) for _ in range(H)]
            all_H_dir: list[list[ibis.expr.types.NumericValue]] = []

            for t in range(len(x_seq) // I):
                x_t = x_seq[t * I : (t + 1) * I]

                gates: list[list[ibis.expr.types.NumericValue]] = []
                for g in range(4):  # IOFC
                    pre = []
                    for h in range(H):
                        w_terms = [x_t[i] * w_val(g, h, i) for i in range(I)]
                        r_terms = [H_st[hid] * r_val(g, h, hid) for hid in range(H)]
                        b_total = bias_in(g, h) + bias_rec(g, h)
                        val = (
                            self._optimizer.fold_operation(sum(w_terms + r_terms))
                            + b_total
                        )
                        pre.append(val)
                    gates.append(pre)

                pre_i, pre_o, pre_f, pre_c = gates  # IOFC

                i_gate = [f_gate(v) for v in pre_i]
                o_gate = [f_gate(v) for v in pre_o]
                f_gate_vals = [f_gate(v) for v in pre_f]
                c_bar = [g_cell(v) for v in pre_c]

                new_C = [
                    self._optimizer.fold_operation(
                        f_gate_vals[h] * C_st[h] + i_gate[h] * c_bar[h]
                    )
                    for h in range(H)
                ]
                new_H = [
                    self._optimizer.fold_operation(o_gate[h] * h_out(new_C[h]))
                    for h in range(H)
                ]

                H_st = new_H
                C_st = new_C
                all_H_dir.append(new_H)

            return all_H_dir, H_st, C_st

        # ── Run direction(s) ──────────────────────────────────────────────
        outputs = self.outputs  # may have 1, 2, or 3 entries; some may be ""

        if direction == "reverse":
            x_rev: list = []
            for t in reversed(range(T)):
                x_rev.extend(x_exprs[t * I : (t + 1) * I])
            all_H_rev, H_state_rev, C_state_rev = _run_direction(
                0, x_rev, _act_gate, _act_cell, _act_out
            )
            # Re-align so Y[t] corresponds to timestep t of the original sequence.
            all_H = list(reversed(all_H_rev))
            _write_sequence_outputs(self._variables, outputs, all_H, H_state_rev)
            if len(outputs) > 2 and outputs[2]:
                self._variables[outputs[2]] = ValueVariablesGroup(
                    {f"out_Yc_{h}": C_state_rev[h] for h in range(H)}
                )
        elif direction == "bidirectional":
            all_H_fwd, H_state_fwd, C_state_fwd = _run_direction(
                0, x_exprs, _act_gate, _act_cell, _act_out
            )
            x_bwd: list = []
            for t in reversed(range(T)):
                x_bwd.extend(x_exprs[t * I : (t + 1) * I])
            all_H_bwd_rev, H_state_bwd, C_state_bwd = _run_direction(
                1, x_bwd, _act_gate_bwd, _act_cell_bwd, _act_out_bwd
            )
            # Re-align: step 0 of the backward pass processed t=T-1, etc.
            all_H_bwd = list(reversed(all_H_bwd_rev))

            _write_bidir_sequence_outputs(
                self._variables, outputs,
                all_H_fwd, all_H_bwd, H_state_fwd, H_state_bwd,
            )
            if len(outputs) > 2 and outputs[2]:
                self._variables[outputs[2]] = ValueVariablesGroup(
                    {
                        **{f"out_Yc_0_{h}": C_state_fwd[h] for h in range(H)},
                        **{f"out_Yc_1_{h}": C_state_bwd[h] for h in range(H)},
                    }
                )
        else:  # forward
            all_H_fwd, H_state_fwd, C_state_fwd = _run_direction(
                0, x_exprs, _act_gate, _act_cell, _act_out
            )
            _write_sequence_outputs(self._variables, outputs, all_H_fwd, H_state_fwd)
            if len(outputs) > 2 and outputs[2]:
                self._variables[outputs[2]] = ValueVariablesGroup(
                    {f"out_Yc_{h}": C_state_fwd[h] for h in range(H)}
                )
