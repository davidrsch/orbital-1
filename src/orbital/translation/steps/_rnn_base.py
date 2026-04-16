"""Shared helpers for ONNX recurrent-unit translators (LSTM, GRU, RNN)."""

import ibis

from ..variables import ValueVariablesGroup


def _sigmoid(v: ibis.expr.types.NumericValue) -> ibis.expr.types.NumericValue:
    """Element-wise sigmoid: 1 / (1 + exp(-v))."""
    return ibis.literal(1.0) / (ibis.literal(1.0) + (-v).exp())


def _resolve_rnn_activation(
    name: str,
    alpha: "float | None" = None,
    beta: "float | None" = None,
):
    """Resolve an ONNX RNN activation name to a callable ``f(v) -> ibis_expr``.

    Supported activations: Sigmoid, Tanh, Relu, HardSigmoid, Elu, LeakyRelu,
    Softsign, Softplus, Affine.

    Parameters
    ----------
    name:
        ONNX activation name (case-insensitive).
    alpha:
        Optional scaling parameter used by some activations (e.g. HardSigmoid,
        Elu, LeakyRelu).  Pass ``None`` to use the ONNX default.
    beta:
        Optional shift parameter used by some activations (e.g. HardSigmoid,
        Affine).  Pass ``None`` to use the ONNX default.
    """
    from .tanh import _tanh  # local import to avoid circular dependency at module load

    name_lower = name.lower()
    if name_lower == "sigmoid":
        return _sigmoid
    elif name_lower == "tanh":
        return _tanh
    elif name_lower == "relu":
        def _relu(v):
            return ibis.greatest(v, ibis.literal(0.0))
        return _relu
    elif name_lower == "hardsigmoid":
        a = float(alpha) if alpha is not None else 0.2
        b = float(beta) if beta is not None else 0.5
        def _hardsig(v):
            return ibis.greatest(
                ibis.literal(0.0),
                ibis.least(ibis.literal(1.0), ibis.literal(a) * v + ibis.literal(b)),
            )
        return _hardsig
    elif name_lower == "leakyrelu":
        a = float(alpha) if alpha is not None else 0.01
        def _leaky(v):
            return ibis.ifelse(v >= ibis.literal(0.0), v, ibis.literal(a) * v)
        return _leaky
    elif name_lower == "elu":
        a = float(alpha) if alpha is not None else 1.0
        def _elu(v):
            return ibis.ifelse(
                v >= ibis.literal(0.0),
                v,
                ibis.literal(a) * (v.exp() - ibis.literal(1.0)),
            )
        return _elu
    elif name_lower == "softsign":
        def _softsign(v):
            return v / (ibis.literal(1.0) + v.abs())
        return _softsign
    elif name_lower == "softplus":
        def _softplus(v):
            return (ibis.literal(1.0) + v.exp()).ln()
        return _softplus
    elif name_lower == "affine":
        a = float(alpha) if alpha is not None else 1.0
        b = float(beta) if beta is not None else 0.0
        def _affine(v):
            return ibis.literal(a) * v + ibis.literal(b)
        return _affine
    else:
        raise NotImplementedError(
            f"RNN activation '{name}' is not supported. "
            "Supported activations: Sigmoid, Tanh, Relu, HardSigmoid, "
            "Elu, LeakyRelu, Softsign, Softplus, Affine."
        )


def _get_flat_weights(
    variables: object,
    input_name: str,
    op_name: str,
) -> tuple[list, list[int]]:
    """Return ``(flat_values, dims)`` for a weight initializer.

    Parameters
    ----------
    variables:
        The ``GraphVariables`` instance (must expose ``get_initializer`` and
        ``get_initializer_value``).
    input_name:
        The ONNX input name that refers to the weight tensor.
    op_name:
        Human-readable operator name used in error messages (e.g. ``"LSTM"``).

    Returns
    -------
    flat_values : list
        1-D list of float values read from the initializer.
    dims : list[int]
        Shape of the tensor (length ≥ 1).

    Raises
    ------
    ValueError
        If the initializer is missing or its values cannot be read.
    """
    tensor = variables.get_initializer(input_name)
    if tensor is None:
        raise ValueError(
            f"{op_name}: weight tensor {input_name!r} not found in initializers."
        )
    flat = variables.get_initializer_value(input_name)
    if flat is None or not isinstance(flat, (list, tuple)):
        raise ValueError(
            f"{op_name}: weight values for {input_name!r} could not be read."
        )
    return list(flat), list(tensor.dims)


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


def _write_bidir_sequence_outputs(
    variables: object,
    output_names: list[str],
    all_H_fwd: list[list],
    all_H_bwd: list[list],
    H_state_fwd: list,
    H_state_bwd: list,
) -> None:
    """Write Y and Y_h outputs for a bidirectional recurrent cell.

    Parameters
    ----------
    variables:
        The ``GraphVariables`` instance on which outputs are set.
    output_names:
        The list of ONNX output names from ``self.outputs``.
    all_H_fwd:
        ``all_H_fwd[t][h]`` — forward hidden state at timestep *t*, unit *h*.
    all_H_bwd:
        ``all_H_bwd[t][h]`` — backward hidden state **at timestep t** (already
        re-aligned so that index *t* corresponds to time *t*, not processing order).
    H_state_fwd:
        Final forward hidden state ``H_state_fwd[h]``.
    H_state_bwd:
        Final backward hidden state ``H_state_bwd[h]``.
    """
    T = len(all_H_fwd)
    H = len(H_state_fwd)

    # Y: full sequence [seq_len, 2, batch, H]
    if output_names and output_names[0]:
        y_group = ValueVariablesGroup(
            {
                **{
                    f"out_Y_{t}_0_{h}": all_H_fwd[t][h]
                    for t in range(T)
                    for h in range(H)
                },
                **{
                    f"out_Y_{t}_1_{h}": all_H_bwd[t][h]
                    for t in range(T)
                    for h in range(H)
                },
            }
        )
        variables[output_names[0]] = y_group

    # Y_h: [2, batch, H]
    if len(output_names) > 1 and output_names[1]:
        y_h_group = ValueVariablesGroup(
            {
                **{f"out_Yh_0_{h}": H_state_fwd[h] for h in range(H)},
                **{f"out_Yh_1_{h}": H_state_bwd[h] for h in range(H)},
            }
        )
        variables[output_names[1]] = y_h_group
