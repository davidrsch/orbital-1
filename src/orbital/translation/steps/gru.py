"""Implementation of the GRU operator for fixed-length sequences."""

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup
from .tanh import _tanh


def _sigmoid(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
    return ibis.literal(1.0) / (ibis.literal(1.0) + (-v).exp())


class GRUTranslator(Translator):
    """Translate the ONNX GRU operator by unrolling the recurrence.

    Limitations (raises :class:`NotImplementedError` otherwise):
    - Forward direction only (``direction="forward"``).
    - Default activations only (Sigmoid/Tanh).
    - Linear-before-reset gate order (ONNX default; ``linear_before_reset=0``).
    - Sequence length ``T`` must be statically recoverable from the input
      group size (``len(X_group) // input_size``).

    ONNX gate order (ZRH):
    - z gate (update) → rows ``[0 : H]``
    - r gate (reset)  → rows ``[H : 2H]``
    - h gate (hidden) → rows ``[2H : 3H]``

    ONNX ``B`` layout ``[1, 6H]``:
    - ``[0 : 3H]``  → input biases   (ZRH)
    - ``[3H : 6H]`` → recurrent biases (ZRH)

    Gate equations (ONNX default, ``linear_before_reset=0``)::

        z_t = sigmoid(X_t @ Wz.T + H_prev @ Rz.T + Wbz + Rbz)
        r_t = sigmoid(X_t @ Wr.T + H_prev @ Rr.T + Wbr + Rbr)
        h_t = tanh(X_t @ Wh.T + (r_t ⊙ H_prev) @ Rh.T + Wbh + Rbh)
        H_t = (1 − z_t) ⊙ h_t + z_t ⊙ H_prev
    """

    _GATE_Z = 0
    _GATE_R = 1
    _GATE_H = 2

    def process(self) -> None:
        """Translate the GRU node, writing result(s) to the graph."""
        # https://onnx.ai/onnx/operators/onnx__GRU.html

        direction = str(self._attributes.get("direction", "forward"))
        if direction != "forward":
            raise NotImplementedError("GRU: only direction='forward' is supported.")

        activations = self._attributes.get("activations", None)
        if activations and list(activations) not in (
            ["Sigmoid", "Tanh"],
            ["sigmoid", "tanh"],
        ):
            raise NotImplementedError(
                "GRU: only default activations [Sigmoid, Tanh] are supported."
            )

        linear_before_reset = int(self._attributes.get("linear_before_reset", 0))
        if linear_before_reset != 0:
            raise NotImplementedError(
                "GRU: linear_before_reset=1 is not supported."
            )

        hidden_size = int(self._attributes["hidden_size"])
        H = hidden_size

        # ── Extract W [1, 3H, I] ──────────────────────────────────────────
        w_tensor = self._variables.get_initializer(self.inputs[1])
        if w_tensor is None:
            raise ValueError("GRU: weight tensor W not found in initializers.")
        _, three_h, I = list(w_tensor.dims)
        if three_h != 3 * H:
            raise ValueError(f"GRU: W dim[1]={three_h} expected 3*hidden_size={3*H}.")

        w_flat = self._variables.get_initializer_value(self.inputs[1])
        if w_flat is None or not isinstance(w_flat, (list, tuple)):
            raise ValueError("GRU: W values could not be read.")

        # ── Extract R [1, 3H, H] ──────────────────────────────────────────
        r_flat = self._variables.get_initializer_value(self.inputs[2])
        if r_flat is None or not isinstance(r_flat, (list, tuple)):
            raise ValueError("GRU: R values could not be read.")

        # ── Extract B [1, 6H] (optional) ─────────────────────────────────
        has_bias = len(self.inputs) > 3 and bool(self.inputs[3])
        b_flat: list = []
        if has_bias:
            b_val = self._variables.get_initializer_value(self.inputs[3])
            if b_val is not None and isinstance(b_val, (list, tuple)):
                b_flat = list(b_val)

        # ── Sequence length and initial state guards ──────────────────────
        # ONNX GRU inputs: [X, W, R, B, sequence_lens, initial_h]
        if len(self.inputs) > 4 and bool(self.inputs[4]):
            raise NotImplementedError(
                "GRU: sequence_lens (input[4]) is not supported; "
                "all sequences must have the same fixed length T."
            )
        if len(self.inputs) > 5 and bool(self.inputs[5]):
            raise NotImplementedError(
                "GRU: non-zero initial_h (input[5]) is not supported; "
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
                f"GRU: input group has {total_in} elements which is not divisible"
                f" by input_size={I}; cannot infer sequence length T."
            )
        T = total_in // I

        # W[g, h, i] → w_flat[g * H * I + h * I + i]
        def w_val(g: int, h: int, i: int) -> float:
            return float(w_flat[g * H * I + h * I + i])

        # R[g, h, hid] → r_flat[g * H * H + h * H + hid]
        def r_val(g: int, h: int, hid: int) -> float:
            return float(r_flat[g * H * H + h * H + hid])

        def bias_in(g: int, h: int) -> float:
            return float(b_flat[g * H + h]) if b_flat else 0.0

        def bias_rec(g: int, h: int) -> float:
            return float(b_flat[3 * H + g * H + h]) if b_flat else 0.0

        # ── Unroll T timesteps ────────────────────────────────────────────
        H_state: list[ibis.expr.types.NumericValue] = [
            ibis.literal(0.0) for _ in range(H)
        ]

        all_H: list[list[ibis.expr.types.NumericValue]] = []

        for t in range(T):
            x_t = x_exprs[t * I : (t + 1) * I]

            # z gate (update gate)
            pre_z = [
                self._optimizer.fold_operation(
                    sum(
                        [x_t[i] * w_val(self._GATE_Z, h, i) for i in range(I)]
                        + [H_state[hid] * r_val(self._GATE_Z, h, hid) for hid in range(H)]
                    )
                    + bias_in(self._GATE_Z, h)
                    + bias_rec(self._GATE_Z, h)
                )
                for h in range(H)
            ]

            # r gate (reset gate)
            pre_r = [
                self._optimizer.fold_operation(
                    sum(
                        [x_t[i] * w_val(self._GATE_R, h, i) for i in range(I)]
                        + [H_state[hid] * r_val(self._GATE_R, h, hid) for hid in range(H)]
                    )
                    + bias_in(self._GATE_R, h)
                    + bias_rec(self._GATE_R, h)
                )
                for h in range(H)
            ]

            z_gate = [_sigmoid(v) for v in pre_z]
            r_gate = [_sigmoid(v) for v in pre_r]

            # h̃ gate – r ⊙ H_prev before the recurrent weight multiplication
            # (ONNX default: linear_before_reset=0)
            pre_h = [
                self._optimizer.fold_operation(
                    sum(
                        [x_t[i] * w_val(self._GATE_H, h, i) for i in range(I)]
                        + [
                            (r_gate[hid] * H_state[hid]) * r_val(self._GATE_H, h, hid)
                            for hid in range(H)
                        ]
                    )
                    + bias_in(self._GATE_H, h)
                    + bias_rec(self._GATE_H, h)
                )
                for h in range(H)
            ]

            h_tilde = [_tanh(v) for v in pre_h]

            new_H = [
                self._optimizer.fold_operation(
                    (ibis.literal(1.0) - z_gate[h]) * h_tilde[h]
                    + z_gate[h] * H_state[h]
                )
                for h in range(H)
            ]

            H_state = new_H
            all_H.append(new_H)

        # ── Set outputs ───────────────────────────────────────────────────
        outputs = self.outputs  # may have 1 or 2 entries; some may be ""

        # Y: full sequence [seq_len, num_directions, batch, H]
        if outputs and outputs[0]:
            y_group = ValueVariablesGroup(
                {
                    f"out_Y_{t}_{h}": all_H[t][h]
                    for t in range(T)
                    for h in range(H)
                }
            )
            self._variables[outputs[0]] = y_group

        # Y_h: final hidden state [num_directions, batch, H]
        if len(outputs) > 1 and outputs[1]:
            y_h_group = ValueVariablesGroup(
                {f"out_Yh_{h}": H_state[h] for h in range(H)}
            )
            self._variables[outputs[1]] = y_h_group
