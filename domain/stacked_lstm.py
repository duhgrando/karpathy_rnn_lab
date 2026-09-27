"""Forward propagation for a stack of LSTM layers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from domain.lstm_model import sigmoid


@dataclass(frozen=True)
class StackedLSTMLayerParams:
    Wxi: np.ndarray
    Whi: np.ndarray
    bi: np.ndarray
    Wxf: np.ndarray
    Whf: np.ndarray
    bf: np.ndarray
    Wxo: np.ndarray
    Who: np.ndarray
    bo: np.ndarray
    Wxg: np.ndarray
    Whg: np.ndarray
    bg: np.ndarray


@dataclass(frozen=True)
class StackedLSTMParams:
    layers: Tuple[StackedLSTMLayerParams, ...]
    Why: np.ndarray
    by: np.ndarray


def initialize_stacked_lstm_parameters(
    vocab_size: int, hidden_sizes: Sequence[int], seed: int = 0
) -> StackedLSTMParams:
    """Initialize bottom-to-top LSTM layers with forget-gate bias one."""
    _validate_hidden_sizes(hidden_sizes)

    rng = np.random.default_rng(seed)

    def weight(rows: int, columns: int) -> np.ndarray:
        return rng.standard_normal((rows, columns)) * 0.01

    layers = []
    input_size = vocab_size
    for hidden_size in hidden_sizes:
        layers.append(
            StackedLSTMLayerParams(
                Wxi=weight(hidden_size, input_size),
                Whi=weight(hidden_size, hidden_size),
                bi=np.zeros((hidden_size, 1)),
                Wxf=weight(hidden_size, input_size),
                Whf=weight(hidden_size, hidden_size),
                bf=np.ones((hidden_size, 1)),
                Wxo=weight(hidden_size, input_size),
                Who=weight(hidden_size, hidden_size),
                bo=np.zeros((hidden_size, 1)),
                Wxg=weight(hidden_size, input_size),
                Whg=weight(hidden_size, hidden_size),
                bg=np.zeros((hidden_size, 1)),
            )
        )
        input_size = hidden_size
    return StackedLSTMParams(
        layers=tuple(layers),
        Why=weight(vocab_size, input_size),
        by=np.zeros((vocab_size, 1)),
    )


def _validate_hidden_sizes(hidden_sizes: Sequence[int]) -> None:
    if not hidden_sizes:
        raise ValueError("hidden_sizes must contain at least one layer")
    if any(size <= 0 for size in hidden_sizes):
        raise ValueError("hidden sizes must be positive")


def compute_layer_gates(
    params: StackedLSTMLayerParams,
    x_t: np.ndarray,
    h_prev: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    input_gate = sigmoid(params.Wxi @ x_t + params.Whi @ h_prev + params.bi)
    forget_gate = sigmoid(params.Wxf @ x_t + params.Whf @ h_prev + params.bf)
    output_gate = sigmoid(params.Wxo @ x_t + params.Who @ h_prev + params.bo)
    candidate = np.tanh(params.Wxg @ x_t + params.Whg @ h_prev + params.bg)
    return input_gate, forget_gate, output_gate, candidate


def _forward_layer(
    params: StackedLSTMLayerParams,
    x_t: np.ndarray,
    h_prev: np.ndarray,
    c_prev: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    input_gate, forget_gate, output_gate, candidate = compute_layer_gates(
        params, x_t, h_prev
    )
    c_t = forget_gate * c_prev + input_gate * candidate
    return output_gate * np.tanh(c_t), c_t


def _validate_forward_inputs(params, h0s, c0s, dropout_probability):
    if len(h0s) != len(params.layers) or len(c0s) != len(params.layers):
        raise ValueError("one initial hidden and cell state is required per layer")
    if not 0.0 <= dropout_probability < 1.0:
        raise ValueError("dropout_probability must be in [0, 1)")


def _inter_layer_input(hidden, dropout_probability, training, seed):
    if not training or dropout_probability == 0.0:
        mask = np.ones_like(hidden)
    else:
        rng = np.random.default_rng(seed)
        mask = rng.binomial(1, 1.0 - dropout_probability, size=hidden.shape)
        mask = mask / (1.0 - dropout_probability)
    return hidden * mask, mask


def _forward_timestep(params, x_t, hs_prev, cs_prev, dropout_probability, training, seed):
    layer_input = x_t
    next_hidden = []
    next_cell = []
    masks = []
    last_layer = len(params.layers) - 1
    for layer_index, (layer, h_prev, c_prev) in enumerate(
        zip(params.layers, hs_prev, cs_prev)
    ):
        h_t, c_t = _forward_layer(layer, layer_input, h_prev, c_prev)
        next_hidden.append(h_t)
        next_cell.append(c_t)
        layer_input = h_t
        if layer_index < last_layer:
            dropout_seed = seed + layer_index if seed is not None else None
            layer_input, mask = _inter_layer_input(
                h_t, dropout_probability, training, dropout_seed
            )
            masks.append(mask)
    logits = params.Why @ layer_input + params.by
    return tuple(next_hidden), tuple(next_cell), logits, _softmax_columns(logits), tuple(masks)


def _initialize_histories(params, h0s, c0s):
    hidden_histories = [[state] for state in h0s]
    cell_histories = [[state] for state in c0s]
    masks = [[] for _ in range(len(params.layers) - 1)]
    return hidden_histories, cell_histories, masks


def _append_timestep(hidden_histories, cell_histories, masks_by_connection, hs, cs, masks):
    updated_hidden = tuple(
        tuple(states) + (hidden,)
        for states, hidden in zip(hidden_histories, hs)
    )
    updated_cell = tuple(
        tuple(states) + (cell,)
        for states, cell in zip(cell_histories, cs)
    )
    updated_masks = tuple(
        tuple(connection_masks) + (mask,)
        for connection_masks, mask in zip(masks_by_connection, masks)
    )
    return updated_hidden, updated_cell, updated_masks


def _freeze_histories(hidden_histories, cell_histories, masks_by_connection):
    return (
        tuple(tuple(states) for states in hidden_histories),
        tuple(tuple(states) for states in cell_histories),
        tuple(tuple(masks) for masks in masks_by_connection),
    )


def _last_states(histories):
    return tuple(states[-1] for states in histories)


def _timestep_dropout_seed(seed, timestep, layer_count):
    if seed is None:
        return None
    return seed + timestep * layer_count


def _softmax_columns(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=0, keepdims=True)


def stacked_lstm_step(
    params: StackedLSTMParams,
    x_t: np.ndarray,
    hs_prev: Sequence[np.ndarray],
    cs_prev: Sequence[np.ndarray],
) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """Advance every layer for one timestep; columns are independent examples."""
    hidden, cell, logits, probabilities, _ = _forward_timestep(
        params, x_t, hs_prev, cs_prev, 0.0, False, None
    )
    return hidden, cell, logits, probabilities


def stacked_lstm_forward_sequence(
    params: StackedLSTMParams,
    inputs: Sequence[np.ndarray],
    h0s: Sequence[np.ndarray],
    c0s: Sequence[np.ndarray],
    dropout_probability: float = 0.0,
    training: bool = False,
    seed: int | None = None,
):
    """Unroll the stack and return states, outputs, probabilities, and masks.

    State arrays use ``(hidden_size, batch_size)`` columns. The returned
    dropout masks are indexed by connection (lower layer) then timestep.
    """
    _validate_forward_inputs(params, h0s, c0s, dropout_probability)

    hs_by_layer, cs_by_layer, masks_by_connection = _initialize_histories(
        params, h0s, c0s
    )
    ys = []
    ps = []

    for timestep, x_t in enumerate(inputs):
        dropout_seed = _timestep_dropout_seed(seed, timestep, len(params.layers))
        current_hs, current_cs, logits, probabilities, masks = _forward_timestep(
            params,
            x_t,
            _last_states(hs_by_layer),
            _last_states(cs_by_layer),
            dropout_probability,
            training,
            dropout_seed,
        )
        ys.append(logits)
        ps.append(probabilities)
        hs_by_layer, cs_by_layer, masks_by_connection = _append_timestep(
            hs_by_layer, cs_by_layer, masks_by_connection, current_hs, current_cs, masks
        )

    frozen_hs, frozen_cs, frozen_masks = _freeze_histories(
        hs_by_layer, cs_by_layer, masks_by_connection
    )
    return (
        frozen_hs,
        frozen_cs,
        tuple(ys),
        tuple(ps),
        frozen_masks,
    )