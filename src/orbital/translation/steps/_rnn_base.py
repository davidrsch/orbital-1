"""Shared helpers for ONNX recurrent-unit translators (LSTM, GRU, RNN)."""

import ibis

from ..variables import ValueVariablesGroup


def _sigmoid(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
    """Element-wise sigmoid: 1 / (1 + exp(-v))."""
    return ibis.literal(1.0) / (ibis.literal(1.0) + (-v).exp())


def _write_sequence_outputs(
    variables: object,
    output_names: list[str],
    all_H: list[list[ibis.expr.types.NumericValue]],
    H_state: list[ibis.expr.types.NumericValue],
) -> None:
    """Write the Y and Y_h outputs shared by LSTM, GRU, and RNN.

    Parameters
    ----------
    variables:
        The ``GraphVariables`` (or equivalent mapping) on which outputs are set.
    output_names:
        The list of ONNX output names from ``self.outputs``.  May be shorter
        than 2 entries; individual entries may be empty strings (unused).
    all_H:
        ``all_H[t][h]`` — the hidden-state expression for timestep *t*,
        unit *h*.
    H_state:
        Final hidden state: ``H_state[h]`` is the expression for unit *h*
        after the last timestep.
    """
    T = len(all_H)
    H = len(H_state)

    # Y: full sequence [seq_len, num_directions, batch, H]
    if output_names and output_names[0]:
        y_group = ValueVariablesGroup(
            {
                f"out_Y_{t}_{h}": all_H[t][h]
                for t in range(T)
                for h in range(H)
            }
        )
        variables[output_names[0]] = y_group

    # Y_h: final hidden state [num_directions, batch, H]
    if len(output_names) > 1 and output_names[1]:
        y_h_group = ValueVariablesGroup(
            {f"out_Yh_{h}": H_state[h] for h in range(H)}
        )
        variables[output_names[1]] = y_h_group
