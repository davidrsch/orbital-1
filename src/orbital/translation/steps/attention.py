"""Implementation of the ONNX ``com.microsoft.Attention`` contrib operator.

This is the *fused QKV* attention op used by many Hugging Face / ONNX-RT
exported transformers. It differs from ``MultiHeadAttention`` in that the
query, key, and value projections share a single packed weight matrix.

Operator inputs
---------------
0  input   : sequence of shape [batch, T, I] — represented as a flat
             ``VariablesGroup`` with T * I columns (time-step major).
1  weight  : initializer of shape [I, 3 * H] — packed QKV weights
             (Q = weight[:, :H], K = weight[:, H:2H], V = weight[:, 2H:3H]).
2  bias    : (optional) initializer of shape [3 * H]
3+ anything beyond input[2] is currently not supported.

Attributes
----------
num_heads   Required.
hidden_size Optional; inferred as H = weight.shape[1] // 3 when absent.

Notes
-----
* No causal masking.
* No past key/value state.
* mask_index (input[3]) and attention_bias (input[5]) are not supported.
"""

import math

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup
from ._rnn_base import _get_flat_weights


class AttentionTranslator(Translator):
    """Translate the ONNX ``com.microsoft.Attention`` operator.

    The fused-QKV attention projection is equivalent to three separate Gemm
    operations followed by scaled dot-product attention over all heads.
    """

    def process(self) -> None:
        """Translate the Attention node, writing the output to the graph."""
        # https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md

        num_heads = int(self._attributes.get("num_heads", 1))

        # ── Consume input sequence ────────────────────────────────────────
        in_val = self._variables.consume(self.inputs[0])
        in_exprs: list = (
            list(in_val.values()) if isinstance(in_val, VariablesGroup) else [in_val]
        )

        # ── Packed weight [I, 3*H] ────────────────────────────────────────
        w_flat, w_dims = _get_flat_weights(self._variables, self.inputs[1], "Attention")
        if len(w_dims) != 2:
            raise ValueError(
                f"Attention: weight tensor must be 2-D, got dims={w_dims}."
            )
        I_size, three_h = w_dims

        # ── Infer hidden size H ───────────────────────────────────────────
        h_attr = self._attributes.get("hidden_size", None)
        if h_attr is not None:
            H = int(h_attr)
            if three_h != 3 * H:
                raise ValueError(
                    f"Attention: weight dim[1]={three_h} inconsistent with "
                    f"hidden_size={H} (expected 3*H={3 * H})."
                )
        else:
            if three_h % 3 != 0:
                raise ValueError(
                    f"Attention: weight dim[1]={three_h} must be divisible by 3."
                )
            H = three_h // 3

        if H % num_heads != 0:
            raise ValueError(
                f"Attention: hidden_size={H} must be divisible by num_heads={num_heads}."
            )
        head_dim = H // num_heads

        # ── Check input size divisibility ─────────────────────────────────
        if len(in_exprs) % I_size != 0:
            raise ValueError(
                f"Attention: input group has {len(in_exprs)} elements, which is "
                f"not divisible by input_size={I_size}."
            )
        T = len(in_exprs) // I_size

        # ── Optional bias [3*H] ───────────────────────────────────────────
        has_bias = len(self.inputs) > 2 and bool(self.inputs[2])
        bias_flat: list[float] = []
        if has_bias:
            b_raw = self._variables.get_initializer_value(self.inputs[2])
            if b_raw is not None and isinstance(b_raw, (list, tuple)):
                bias_flat = [float(v) for v in b_raw]

        # ── Unsupported optional inputs ───────────────────────────────────
        if len(self.inputs) > 3 and bool(self.inputs[3]):
            raise NotImplementedError(
                "Attention: mask_index (input[3]) is not supported."
            )
        if len(self.inputs) > 4 and bool(self.inputs[4]):
            raise NotImplementedError("Attention: past (input[4]) is not supported.")
        if len(self.inputs) > 5 and bool(self.inputs[5]):
            raise NotImplementedError(
                "Attention: attention_bias (input[5]) is not supported."
            )
        # w_flat layout: [I, 3*H] in row-major order.
        # w_flat[i * 3*H + j] = weight[i, j]
        # Q slice: j in [0, H), K slice: j in [H, 2H), V slice: j in [2H, 3H)

        def _proj(t: int, j_start: int, j_end: int) -> list:
            """Compute linear projection of in_exprs[t*I_size:(t+1)*I_size] using columns [j_start, j_end)."""
            x_t = in_exprs[t * I_size : (t + 1) * I_size]
            return [
                self._optimizer.fold_operation(
                    sum(x_t[i] * float(w_flat[i * three_h + j]) for i in range(I_size))
                    + (bias_flat[j] if bias_flat else 0.0)
                )
                for j in range(j_start, j_end)
            ]

        # ── Project Q, K, V ───────────────────────────────────────────────
        # Each is a flat list of T*H expressions, indexed as [t*H + d]
        q_proj = [v for t in range(T) for v in _proj(t, 0, H)]
        k_proj = [v for t in range(T) for v in _proj(t, H, 2 * H)]
        v_proj = [v for t in range(T) for v in _proj(t, 2 * H, 3 * H)]

        scale = ibis.literal(1.0 / math.sqrt(head_dim))

        # ── Scaled dot-product attention per head ─────────────────────────
        out_exprs: dict[str, ibis.expr.types.Value] = {}

        for h_idx in range(num_heads):
            h_off = h_idx * head_dim

            # Scores: (T, T) matrix — score[q_i][k_j]
            scores: list[list] = []
            for q_i in range(T):
                row = []
                for k_j in range(T):
                    dot = self._optimizer.fold_operation(
                        sum(
                            q_proj[q_i * H + h_off + d] * k_proj[k_j * H + h_off + d]
                            for d in range(head_dim)
                        )
                    )
                    row.append(scale * dot)
                scores.append(row)

            # Max-stabilised softmax over k_j  ← numerically safe
            softmax_weights: list[list] = []
            for q_i in range(T):
                score_row = scores[q_i]
                if len(score_row) == 1:
                    max_s = score_row[0]
                else:
                    max_s = ibis.greatest(*score_row)
                exp_s = [
                    self._optimizer.fold_operation((s - max_s).exp()) for s in score_row
                ]
                if len(exp_s) == 1:
                    sum_e = exp_s[0]
                else:
                    sum_e = self._optimizer.fold_operation(sum(exp_s))
                softmax_weights.append([e / sum_e for e in exp_s])

            # Output: out[q_i, h_idx * head_dim + d_v] = Σ_k softmax[q_i,k] * V[k, h_off+d_v]
            for q_i in range(T):
                for d_v in range(head_dim):
                    out_col = h_idx * head_dim + d_v
                    out_val = self._optimizer.fold_operation(
                        sum(
                            softmax_weights[q_i][k_j] * v_proj[k_j * H + h_off + d_v]
                            for k_j in range(T)
                        )
                    )
                    out_exprs[f"out_attn_{q_i}_{out_col}"] = out_val

        self._variables[self.outputs[0]] = ValueVariablesGroup(out_exprs)
