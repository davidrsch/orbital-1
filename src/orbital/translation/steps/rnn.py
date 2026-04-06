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
from ._rnn_base import _write_sequence_outputs
from .tanh import _tanh


class RNNTranslator(Translator):
    """Translate the ONNX RNN operator by unrolling the recurrence."""

    def process(self) -> None:
        """Translate the RNN node, writing result(s) to the graph."""
        # https://onnx.ai/onnx/operators/onnx__RNN.html

        direction = str(self._attributes.get("direction", "forward"))
        if direction != "forward":
            raise NotImplementedError("RNN: only direction='forward' is supported.")

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
        w_tensor = self._variables.get_initializer(self.inputs[1])
        if w_tensor is None:
            raise ValueError("RNN: weight tensor W not found in initializers.")
        _, H_check, I = list(w_tensor.dims)
        if H_check != H:
            raise ValueError(f"RNN: W dim[1]={H_check} expected hidden_size={H}.")

        w_flat = self._variables.get_initializer_value(self.inputs[1])
        if w_flat is None or not isinstance(w_flat, (list, tuple)):
            raise ValueError("RNN: W values could not be read.")

        # ── Extract R [1, H, H] ───────────────────────────────────────────
        r_flat = self._variables.get_initializer_value(self.inputs[2])
        if r_flat is None or not isinstance(r_flat, (list, tuple)):
            raise ValueError("RNN: R values could not be read.")

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

        # W[h, i] → w_flat[h * I + i]
        def w_val(h: int, i: int) -> float:
            return float(w_flat[h * I + i])

        # R[h, hid] → r_flat[h * H + hid]
        def r_val(h: int, hid: int) -> float:
            return float(r_flat[h * H + hid])

        def bias_in(h: int) -> float:
            return float(b_flat[h]) if b_flat else 0.0

        def bias_rec(h: int) -> float:
            return float(b_flat[H + h]) if b_flat else 0.0

        # ── Unroll T timesteps ────────────────────────────────────────────
        H_state: list[ibis.expr.types.NumericValue] = [
            ibis.literal(0.0) for _ in range(H)
        ]

        all_H: list[list[ibis.expr.types.NumericValue]] = []

        for t in range(T):
            x_t = x_exprs[t * I : (t + 1) * I]

            pre_h = [
                self._optimizer.fold_operation(
                    sum(
                        [x_t[i] * w_val(h, i) for i in range(I)]
                        + [H_state[hid] * r_val(h, hid) for hid in range(H)]
                    )
                    + bias_in(h)
                    + bias_rec(h)
                )
                for h in range(H)
            ]

            new_H = [_tanh(v) for v in pre_h]
            H_state = new_H
            all_H.append(new_H)

        # ── Set outputs ───────────────────────────────────────────────────
        outputs = self.outputs  # may have 1 or 2 entries; some may be ""

        _write_sequence_outputs(self._variables, outputs, all_H, H_state)
