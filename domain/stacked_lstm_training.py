"""Backpropagation and optimizer state for stacked LSTM layers."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence, Tuple

import numpy as np

from domain.optimization import adagrad_step
from domain.stacked_lstm import (
    StackedLSTMLayerParams,
    StackedLSTMParams,
    compute_layer_gates,
)


@dataclass(frozen=True)
class StackedLSTMLayerGradients:
    dWxi: np.ndarray
    dWhi: np.ndarray
    dbi: np.ndarray
    dWxf: np.ndarray
    dWhf: np.ndarray
    dbf: np.ndarray
    dWxo: np.ndarray
    dWho: np.ndarray
    dbo: np.ndarray
    dWxg: np.ndarray
    dWhg: np.ndarray
    dbg: np.ndarray


@dataclass(frozen=True)
class StackedLSTMGradients:
    layers: Tuple[StackedLSTMLayerGradients, ...]
    dWhy: np.ndarray
    dby: np.ndarray


@dataclass(frozen=True)
class StackedLSTMLayerMemory:
    mWxi: np.ndarray
    mWhi: np.ndarray
    mbi: np.ndarray
    mWxf: np.ndarray
    mWhf: np.ndarray
    mbf: np.ndarray
    mWxo: np.ndarray
    mWho: np.ndarray
    mbo: np.ndarray
    mWxg: np.ndarray
    mWhg: np.ndarray
    mbg: np.ndarray


@dataclass(frozen=True)
class StackedLSTMMemory:
    layers: Tuple[StackedLSTMLayerMemory, ...]
    mWhy: np.ndarray
    mby: np.ndarray


def zero_stacked_lstm_memory(params: StackedLSTMParams) -> StackedLSTMMemory:
    layer_memories = tuple(
        StackedLSTMLayerMemory(
            **{
                f"m{field.name}": np.zeros_like(getattr(layer, field.name))
                for field in fields(layer)
            }
        )
        for layer in params.layers
    )
    return StackedLSTMMemory(
        layers=layer_memories,
        mWhy=np.zeros_like(params.Why),
        mby=np.zeros_like(params.by),
    )


def clip_stacked_lstm_gradients(
    gradients: StackedLSTMGradients, clip: float = 5.0
) -> StackedLSTMGradients:
    clipped_layers = tuple(
        StackedLSTMLayerGradients(
            **{
                field.name: np.clip(getattr(layer, field.name), -clip, clip)
                for field in fields(layer)
            }
        )
        for layer in gradients.layers
    )
    return StackedLSTMGradients(
        layers=clipped_layers,
        dWhy=np.clip(gradients.dWhy, -clip, clip),
        dby=np.clip(gradients.dby, -clip, clip),
    )


def stacked_lstm_adagrad_update(
    params: StackedLSTMParams,
    gradients: StackedLSTMGradients,
    memory: StackedLSTMMemory,
    learning_rate: float = 0.1,
) -> Tuple[StackedLSTMParams, StackedLSTMMemory]:
    updated_layers = []
    updated_memories = []
    for layer, layer_gradients, layer_memory in zip(
        params.layers, gradients.layers, memory.layers
    ):
        updated_params = {}
        updated_memory = {}
        for field in fields(layer):
            name = field.name
            new_param, new_accumulator = adagrad_step(
                getattr(layer, name),
                getattr(layer_gradients, f"d{name}"),
                getattr(layer_memory, f"m{name}"),
                learning_rate,
            )
            updated_params[name] = new_param
            updated_memory[f"m{name}"] = new_accumulator
        updated_layers.append(StackedLSTMLayerParams(**updated_params))
        updated_memories.append(StackedLSTMLayerMemory(**updated_memory))

    new_why, new_mwhy = adagrad_step(
        params.Why, gradients.dWhy, memory.mWhy, learning_rate
    )
    new_by, new_mby = adagrad_step(
        params.by, gradients.dby, memory.mby, learning_rate
    )
    return (
        StackedLSTMParams(tuple(updated_layers), new_why, new_by),
        StackedLSTMMemory(tuple(updated_memories), new_mwhy, new_mby),
    )


def _target_matrix(targets: Sequence[Sequence[int]], time_steps: int, batch_size: int):
    target_matrix = np.asarray(targets, dtype=np.intp)
    if target_matrix.ndim == 1 and batch_size == 1:
        target_matrix = target_matrix[:, np.newaxis]
    if target_matrix.shape != (time_steps, batch_size):
        raise ValueError("targets must have shape (time_steps, batch_size)")
    return target_matrix


def stacked_lstm_cross_entropy_loss(
    probabilities: Sequence[np.ndarray], targets: Sequence[Sequence[int]]
) -> float:
    """Return mean cross-entropy over all target tokens in a sequence batch."""
    batch_size = probabilities[0].shape[1]
    target_matrix = _target_matrix(targets, len(probabilities), batch_size)
    columns = np.arange(batch_size)
    losses = [
        -np.log(probabilities[timestep][target_matrix[timestep], columns] + 1e-12)
        for timestep in range(len(probabilities))
    ]
    return float(np.mean(losses))


def _zero_layer_gradients(layer):
    return {
        f"d{field.name}": np.zeros_like(getattr(layer, field.name))
        for field in fields(layer)
    }


def _initialize_gradient_accumulators(params: StackedLSTMParams, hs_by_layer):
    layer_gradients = [_zero_layer_gradients(layer) for layer in params.layers]
    dh_next = [np.zeros_like(states[0]) for states in hs_by_layer]
    dc_next = [np.zeros_like(states[0]) for states in hs_by_layer]
    return layer_gradients, dh_next, dc_next


def _output_step_gradients(params, probabilities, targets, top_hidden, scale):
    batch_size = probabilities.shape[1]
    dy = probabilities.copy()
    dy[targets, np.arange(batch_size)] -= 1.0
    dy *= scale
    return dy @ top_hidden.T, np.sum(dy, axis=1, keepdims=True), params.Why.T @ dy


def _layer_step_gradients(
    layer: StackedLSTMLayerParams,
    layer_input: np.ndarray,
    h_prev: np.ndarray,
    c_prev: np.ndarray,
    c_t: np.ndarray,
    dh: np.ndarray,
    dc_next: np.ndarray,
):
    input_gate, forget_gate, output_gate, candidate = compute_layer_gates(
        layer, layer_input, h_prev
    )
    tanh_cell = np.tanh(c_t)
    d_output = dh * tanh_cell * output_gate * (1.0 - output_gate)
    dc = dh * output_gate * (1.0 - tanh_cell ** 2) + dc_next
    d_input = dc * candidate * input_gate * (1.0 - input_gate)
    d_candidate = dc * input_gate * (1.0 - candidate ** 2)
    d_forget = dc * c_prev * forget_gate * (1.0 - forget_gate)
    gradients = {
        "dWxi": d_input @ layer_input.T,
        "dWhi": d_input @ h_prev.T,
        "dbi": np.sum(d_input, axis=1, keepdims=True),
        "dWxf": d_forget @ layer_input.T,
        "dWhf": d_forget @ h_prev.T,
        "dbf": np.sum(d_forget, axis=1, keepdims=True),
        "dWxo": d_output @ layer_input.T,
        "dWho": d_output @ h_prev.T,
        "dbo": np.sum(d_output, axis=1, keepdims=True),
        "dWxg": d_candidate @ layer_input.T,
        "dWhg": d_candidate @ h_prev.T,
        "dbg": np.sum(d_candidate, axis=1, keepdims=True),
    }
    dh_prev = (
        layer.Whi.T @ d_input
        + layer.Whf.T @ d_forget
        + layer.Who.T @ d_output
        + layer.Whg.T @ d_candidate
    )
    dc_prev = dc * forget_gate
    d_layer_input = (
        layer.Wxi.T @ d_input
        + layer.Wxf.T @ d_forget
        + layer.Wxo.T @ d_output
        + layer.Wxg.T @ d_candidate
    )
    return gradients, dh_prev, dc_prev, d_layer_input


def _layer_input_for_backward(inputs, hs_by_layer, dropout_masks, layer_index, timestep):
    if layer_index == 0:
        return inputs[timestep]
    return (
        hs_by_layer[layer_index - 1][timestep + 1]
        * dropout_masks[layer_index - 1][timestep]
    )


def _backpropagate_layer_step(
    params,
    layer_index,
    layer_input,
    hs_by_layer,
    cs_by_layer,
    dropout_masks,
    timestep,
    gradient_from_above,
    dh_next,
    dc_next,
    layer_gradients,
):
    layer_step_gradients, dh_prev, dc_prev, d_layer_input = _layer_step_gradients(
        params.layers[layer_index],
        layer_input,
        hs_by_layer[layer_index][timestep],
        cs_by_layer[layer_index][timestep],
        cs_by_layer[layer_index][timestep + 1],
        gradient_from_above + dh_next[layer_index],
        dc_next[layer_index],
    )
    updated_layer_gradients = {
        name: layer_gradients[layer_index][name] + gradient
        for name, gradient in layer_step_gradients.items()
    }
    if layer_index > 0:
        d_layer_input *= dropout_masks[layer_index - 1][timestep]
    return d_layer_input, dh_prev, dc_prev, updated_layer_gradients


def stacked_lstm_backpropagate_through_time(
    params: StackedLSTMParams,
    inputs: Sequence[np.ndarray],
    targets: Sequence[Sequence[int]],
    hs_by_layer: Sequence[Sequence[np.ndarray]],
    cs_by_layer: Sequence[Sequence[np.ndarray]],
    probabilities: Sequence[np.ndarray],
    dropout_masks: Sequence[Sequence[np.ndarray]],
) -> StackedLSTMGradients:
    """Backpropagate mean token loss through time, depth, and dropout masks."""
    batch_size = inputs[0].shape[1]
    target_matrix = _target_matrix(targets, len(inputs), batch_size)
    scale = 1.0 / (len(inputs) * batch_size)
    layer_gradients, dh_next, dc_next = _initialize_gradient_accumulators(
        params, hs_by_layer
    )
    dWhy = np.zeros_like(params.Why)
    dby = np.zeros_like(params.by)
    columns = np.arange(batch_size)

    for timestep in reversed(range(len(inputs))):
        d_why, d_by, gradient_from_above = _output_step_gradients(
            params,
            probabilities[timestep],
            target_matrix[timestep],
            hs_by_layer[-1][timestep + 1],
            scale,
        )
        dWhy += d_why
        dby += d_by
        for layer_index in reversed(range(len(params.layers))):
            layer_input = _layer_input_for_backward(
                inputs, hs_by_layer, dropout_masks, layer_index, timestep
            )
            (
                gradient_from_above,
                dh_next_value,
                dc_next_value,
                layer_gradients[layer_index],
            ) = (
                _backpropagate_layer_step(
                    params, layer_index, layer_input, hs_by_layer, cs_by_layer,
                    dropout_masks, timestep, gradient_from_above, dh_next,
                    dc_next, layer_gradients,
                )
            )
            dh_next[layer_index] = dh_next_value
            dc_next[layer_index] = dc_next_value

    return StackedLSTMGradients(
        layers=tuple(StackedLSTMLayerGradients(**gradients) for gradients in layer_gradients),
        dWhy=dWhy,
        dby=dby,
    )