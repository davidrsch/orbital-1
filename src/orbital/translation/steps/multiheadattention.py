"""Implementation of the ONNX MultiHeadAttention operator (com.microsoft contrib op).

Supports fixed-length sequences; no causal masking; no past key/value.
The bias input (input[3]) is used to infer the hidden dimension when present,
or the ``hidden_size`` attribute may be set explicitly.
"""

import math

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup


class MultiHeadAttentionTranslator(Translator):
    """Translate the ONNX ``com.microsoft.MultiHeadAttention`` contrib operator.

    Inputs (ONNX convention):
        0 – query  : VariablesGroup with T_q * D elements (time-step major)
        1 – key    : VariablesGroup with T_k * D elements (optional; defaults to query)
        2 – value  : VariablesGroup with T_v * D_v elements (optional; defaults to query)
        3 – bias   : initializer tensor of shape (D + D + D_v,) (optional)

    Attributes:
        num_heads   – required; number of attention heads
        hidden_size – optional; D per timestep; inferred from bias when omitted

    Output:
        0 – output : VariablesGroup with T_q * D_v elements (time-step major)
    """

    def process(self) -> None:
        num_heads = int(self._attributes.get("num_heads", 1))

        # ── Consume Q, K, V ──────────────────────────────────────────────────
        q_val = self._variables.consume(self.inputs[0])
        q_exprs: list = (
            list(q_val.values()) if isinstance(q_val, VariablesGroup) else [q_val]
        )

        has_key = len(self.inputs) > 1 and bool(self.inputs[1])
        has_value = len(self.inputs) > 2 and bool(self.inputs[2])
        has_bias = len(self.inputs) > 3 and bool(self.inputs[3])

        if has_key:
            k_val = self._variables.consume(self.inputs[1])
            k_exprs: list = (
                list(k_val.values()) if isinstance(k_val, VariablesGroup) else [k_val]
            )
        else:
            k_exprs = list(q_exprs)

        if has_value:
            v_val = self._variables.consume(self.inputs[2])
            v_exprs: list = (
                list(v_val.values()) if isinstance(v_val, VariablesGroup) else [v_val]
            )
        else:
            v_exprs = list(q_exprs)

        # ── Infer hidden dimension D ──────────────────────────────────────────
        if has_bias:
            bias_flat_raw = self._variables.get_initializer_value(self.inputs[3])
            if bias_flat_raw is None or not isinstance(bias_flat_raw, (list, tuple)):
                raise ValueError(
                    "MultiHeadAttention: bias initializer could not be read."
                )
            bias_flat = [float(x) for x in bias_flat_raw]
            # bias layout: [Q_bias (D) | K_bias (D) | V_bias (D_v)]
            # Provisional D: assume self-attention (D_v == D) so bias_size = 3*D.
            # The exact validation len(bias_flat) == 2*D + D_v is done after D_v
            # is inferred from the value sequence below.
            D = len(bias_flat) // 3
        else:
            d_attr = self._attributes.get("hidden_size", None)
            if d_attr is None:
                raise NotImplementedError(
                    "MultiHeadAttention: cannot infer hidden size without a bias "
                    "input. Set the 'hidden_size' attribute on the ONNX node "
                    "or supply the bias tensor (input 3)."
                )
            D = int(d_attr)
            bias_flat = None

        if len(q_exprs) % D != 0:
            raise ValueError(
                f"MultiHeadAttention: query group size {len(q_exprs)} is not "
                f"divisible by hidden_size={D}."
            )
        T_q = len(q_exprs) // D
        T_k = len(k_exprs) // D if len(k_exprs) % D == 0 else len(k_exprs)
        T_v = len(v_exprs) // D if len(v_exprs) % D == 0 else len(v_exprs)
        # value feature size per position: infer from V column count
        D_v = len(v_exprs) // T_k if len(v_exprs) % T_k == 0 else D

        if has_bias and bias_flat is not None and len(bias_flat) != 2 * D + D_v:
            raise ValueError(
                f"MultiHeadAttention: bias size {len(bias_flat)} must equal "
                f"2*D + D_v = 2*{D} + {D_v} = {2 * D + D_v}."
            )

        if D % num_heads != 0:
            raise ValueError(
                f"MultiHeadAttention: hidden_size={D} must be divisible "
                f"by num_heads={num_heads}."
            )
        head_dim = D // num_heads
        v_head_dim = D_v // num_heads

        # ── Add Q, K, V biases ────────────────────────────────────────────────
        if bias_flat is not None:
            q_exprs = [
                q_exprs[t * D + d] + ibis.literal(bias_flat[d])
                for t in range(T_q)
                for d in range(D)
            ]
            k_exprs = [
                k_exprs[t * D + d] + ibis.literal(bias_flat[D + d])
                for t in range(T_k)
                for d in range(D)
            ]
            v_exprs = [
                v_exprs[t * D_v + d] + ibis.literal(bias_flat[2 * D + d])
                for t in range(T_v)
                for d in range(D_v)
            ]

        scale = ibis.literal(1.0 / math.sqrt(head_dim))

        # ── Scaled dot-product attention per head ─────────────────────────────
        # Output shape: T_q * D_v, laid out as [q=0,d=0], [q=0,d=1], ..., [q=T_q-1, d=D_v-1]
        # where d is the full output channel index across all heads (h * v_head_dim + d_v).

        out_exprs: dict[str, ibis.expr.types.Value] = {}

        for h in range(num_heads):
            q_offset = h * head_dim
            k_offset = h * head_dim
            v_offset = h * v_head_dim

            # Scores: (T_q, T_k) matrix of ibis expressions
            # score[q_i, k_j] = scale * Σ_d Q[q_i, q_offset+d] * K[k_j, k_offset+d]
            scores: list[list] = []
            for q_i in range(T_q):
                row = []
                for k_j in range(T_k):
                    dot = self._optimizer.fold_operation(
                        sum(
                            q_exprs[q_i * D + q_offset + d]
                            * k_exprs[k_j * D + k_offset + d]
                            for d in range(head_dim)
                        )
                    )
                    row.append(scale * dot)
                scores.append(row)

            # Softmax over k_j per query position (max-stabilised for numeric stability)
            softmax: list[list] = []
            for q_i in range(T_q):
                score_row = scores[q_i]
                if len(score_row) == 1:
                    max_score = score_row[0]
                else:
                    max_score = ibis.greatest(*score_row)
                exp_scores = [
                    self._optimizer.fold_operation((s - max_score).exp())
                    for s in score_row
                ]
                if len(exp_scores) == 1:
                    sum_exp = exp_scores[0]
                else:
                    sum_exp = self._optimizer.fold_operation(sum(exp_scores))
                softmax.append([e / sum_exp for e in exp_scores])

            # Output: out[q_i, h*v_head_dim + d_v] = Σ_k softmax[q_i,k] * V[k, v_offset+d_v]
            for q_i in range(T_q):
                for d_v in range(v_head_dim):
                    out_col = h * v_head_dim + d_v
                    out_val = self._optimizer.fold_operation(
                        sum(
                            softmax[q_i][k_j] * v_exprs[k_j * D_v + v_offset + d_v]
                            for k_j in range(T_k)
                        )
                    )
                    out_key = f"out_mha_{q_i}_{out_col}"
                    out_exprs[out_key] = out_val

        self._variables[self.outputs[0]] = ValueVariablesGroup(out_exprs)
