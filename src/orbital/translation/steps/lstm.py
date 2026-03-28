"""Implementation of the LSTM operator for fixed-length sequences."""

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup
from .tanh import _tanh


def _sigmoid(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
    return ibis.literal(1.0) / (ibis.literal(1.0) + (-v).exp())


class LSTMTranslator(Translator):
    """Translate the ONNX LSTM operator by unrolling the recurrence.

    Limitations (raises :class:`NotImplementedError` otherwise):
    - Forward direction only (``direction="forward"``).
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
        if direction != "forward":
            raise NotImplementedError("LSTM: only direction='forward' is supported.")

        activations = self._attributes.get("activations", None)
        if activations and list(activations) not in (
            ["Sigmoid", "Tanh", "Tanh"],
            ["sigmoid", "tanh", "tanh"],
        ):
            raise NotImplementedError(
                "LSTM: only default activations [Sigmoid, Tanh, Tanh] are supported."
            )

        input_forget = int(self._attributes.get("input_forget", 0))
        if input_forget != 0:
            raise NotImplementedError("LSTM: input_forget=1 (coupled gates) is not supported.")

        hidden_size = int(self._attributes["hidden_size"])
        H = hidden_size

        # ── Extract W [1, 4H, I] ──────────────────────────────────────────
        w_tensor = self._variables.get_initializer(self.inputs[1])
        if w_tensor is None:
            raise ValueError("LSTM: weight tensor W not found in initializers.")
        _, four_h, I = list(w_tensor.dims)
        if four_h != 4 * H:
            raise ValueError(f"LSTM: W dim[1]={four_h} expected 4*hidden_size={4*H}.")

        w_flat = self._variables.get_initializer_value(self.inputs[1])
        if w_flat is None or not isinstance(w_flat, (list, tuple)):
            raise ValueError("LSTM: W values could not be read.")

        # ── Extract R [1, 4H, H] ──────────────────────────────────────────
        r_flat = self._variables.get_initializer_value(self.inputs[2])
        if r_flat is None or not isinstance(r_flat, (list, tuple)):
            raise ValueError("LSTM: R values could not be read.")

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

        # W[g, h, i] → w_flat[g * H * I + h * I + i]
        def w_val(g: int, h: int, i: int) -> float:
            return float(w_flat[g * H * I + h * I + i])

        # R[g, h, hid] → r_flat[g * H * H + h * H + hid]
        def r_val(g: int, h: int, hid: int) -> float:
            return float(r_flat[g * H * H + h * H + hid])

        # B: input bias for gate g, unit h → b_flat[g * H + h]
        # B: recurrent bias for gate g, unit h → b_flat[4*H + g*H + h]
        def bias_in(g: int, h: int) -> float:
            return float(b_flat[g * H + h]) if b_flat else 0.0

        def bias_rec(g: int, h: int) -> float:
            return float(b_flat[4 * H + g * H + h]) if b_flat else 0.0

        # ── Unroll T timesteps ────────────────────────────────────────────
        H_state: list[ibis.expr.types.NumericValue] = [
            ibis.literal(0.0) for _ in range(H)
        ]
        C_state: list[ibis.expr.types.NumericValue] = [
            ibis.literal(0.0) for _ in range(H)
        ]

        # Collect all Y timesteps if output Y is requested
        all_H: list[list[ibis.expr.types.NumericValue]] = []

        for t in range(T):
            x_t = x_exprs[t * I : (t + 1) * I]

            gates: list[list[ibis.expr.types.NumericValue]] = []
            for g in range(4):  # IOFC
                pre = []
                for h in range(H):
                    w_terms = [x_t[i] * w_val(g, h, i) for i in range(I)]
                    r_terms = [H_state[hid] * r_val(g, h, hid) for hid in range(H)]
                    b_total = bias_in(g, h) + bias_rec(g, h)
                    val = (
                        self._optimizer.fold_operation(sum(w_terms + r_terms))
                        + b_total
                    )
                    pre.append(val)
                gates.append(pre)

            pre_i, pre_o, pre_f, pre_c = gates  # IOFC

            i_gate = [_sigmoid(v) for v in pre_i]
            o_gate = [_sigmoid(v) for v in pre_o]
            f_gate = [_sigmoid(v) for v in pre_f]
            c_bar = [_tanh(v) for v in pre_c]

            new_C = [
                self._optimizer.fold_operation(
                    f_gate[h] * C_state[h] + i_gate[h] * c_bar[h]
                )
                for h in range(H)
            ]
            new_H = [
                self._optimizer.fold_operation(o_gate[h] * _tanh(new_C[h]))
                for h in range(H)
            ]

            H_state = new_H
            C_state = new_C
            all_H.append(new_H)

        # ── Set outputs ───────────────────────────────────────────────────
        outputs = self.outputs  # may have 1, 2, or 3 entries; some may be ""

        # Y: full sequence output [seq_len, num_directions, batch, H]
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

        # Y_c: final cell state [num_directions, batch, H]
        if len(outputs) > 2 and outputs[2]:
            y_c_group = ValueVariablesGroup(
                {f"out_Yc_{h}": C_state[h] for h in range(H)}
            )
            self._variables[outputs[2]] = y_c_group
