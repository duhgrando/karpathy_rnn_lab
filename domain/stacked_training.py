"""Domain: backpropagation through a stack of RNN layers.

Each layer's BPTT is exactly domain/training.py's backpropagate_through_time() -- the same
tanh'(h) chain rule -- with one twist: a middle or bottom layer's "loss
gradient" doesn't come from Why/targets directly, it comes from the layer
*above* it (that layer's own dh_raw, pulled back through that layer's own
Wxh). So the backward pass walks time in reverse as usual, and at each time
step walks the stack top-to-bottom, handing a `gradient_from_above` value
down from each layer to the one below it -- the mirror image of how the
forward pass hands each layer's output *up* to the one above it.

Optimizer mechanics reuse domain/optimization.py's adagrad_step() directly
(rather than adagrad_update()'s dataclass-reflection version) because a
stack's parameters are nested (a tuple of per-layer dataclasses plus a top-
level Why/by), not a flat dataclass the naming convention alone can reach.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from domain.optimization import adagrad_step
from domain.stacked_rnn import LayerParams, StackedRNNParams


@dataclass(frozen=True)
class LayerGradients:
    dWxh: np.ndarray
    dWhh: np.ndarray
    dbh: np.ndarray


@dataclass(frozen=True)
class StackedGradients:
    layers: Tuple[LayerGradients, ...]
    dWhy: np.ndarray
    dby: np.ndarray


@dataclass(frozen=True)
class LayerMemory:
    mWxh: np.ndarray
    mWhh: np.ndarray
    mbh: np.ndarray


@dataclass(frozen=True)
class StackedMemory:
    layers: Tuple[LayerMemory, ...]
    mWhy: np.ndarray
    mby: np.ndarray


def zero_stacked_memory(params: StackedRNNParams) -> StackedMemory:
    layer_memories = tuple(
        LayerMemory(
            mWxh=np.zeros_like(layer.Wxh),
            mWhh=np.zeros_like(layer.Whh),
            mbh=np.zeros_like(layer.bh),
        )
        for layer in params.layers
    )
    return StackedMemory(
        layers=layer_memories, mWhy=np.zeros_like(params.Why), mby=np.zeros_like(params.by)
    )


def clip_stacked_gradients(grads: StackedGradients, clip: float = 5.0) -> StackedGradients:
    clipped_layers = tuple(
        LayerGradients(
            dWxh=np.clip(g.dWxh, -clip, clip),
            dWhh=np.clip(g.dWhh, -clip, clip),
            dbh=np.clip(g.dbh, -clip, clip),
        )
        for g in grads.layers
    )
    return StackedGradients(
        layers=clipped_layers,
        dWhy=np.clip(grads.dWhy, -clip, clip),
        dby=np.clip(grads.dby, -clip, clip),
    )


def stacked_adagrad_update(
    params: StackedRNNParams,
    grads: StackedGradients,
    memory: StackedMemory,
    learning_rate: float = 0.1,
) -> Tuple[StackedRNNParams, StackedMemory]:
    new_layers_params, new_layers_memory = [], []
    for layer, layer_grad, layer_mem in zip(params.layers, grads.layers, memory.layers):
        Wxh, mWxh = adagrad_step(layer.Wxh, layer_grad.dWxh, layer_mem.mWxh, learning_rate)
        Whh, mWhh = adagrad_step(layer.Whh, layer_grad.dWhh, layer_mem.mWhh, learning_rate)
        bh, mbh = adagrad_step(layer.bh, layer_grad.dbh, layer_mem.mbh, learning_rate)
        new_layers_params.append(LayerParams(Wxh=Wxh, Whh=Whh, bh=bh))
        new_layers_memory.append(LayerMemory(mWxh=mWxh, mWhh=mWhh, mbh=mbh))

    Why, mWhy = adagrad_step(params.Why, grads.dWhy, memory.mWhy, learning_rate)
    by, mby = adagrad_step(params.by, grads.dby, memory.mby, learning_rate)

    new_params = StackedRNNParams(layers=tuple(new_layers_params), Why=Why, by=by)
    new_memory = StackedMemory(layers=tuple(new_layers_memory), mWhy=mWhy, mby=mby)
    return new_params, new_memory


def _initialize_layer_gradients(params, hs_by_layer):
    layer_grads = [
        {
            "dWxh": np.zeros_like(layer.Wxh),
            "dWhh": np.zeros_like(layer.Whh),
            "dbh": np.zeros_like(layer.bh),
        }
        for layer in params.layers
    ]
    dh_next_by_layer = [np.zeros_like(hs_by_layer[i][0]) for i in range(len(params.layers))]
    return layer_grads, dh_next_by_layer


def _accumulate_layer_gradients(
    params, inputs, hs_by_layer, t, gradient_from_above, layer_grads, dh_next_by_layer
):
    updated_layer_gradients = [dict(gradients) for gradients in layer_grads]
    updated_next_hidden_gradients = list(dh_next_by_layer)
    for layer_index in reversed(range(len(params.layers))):
        layer = params.layers[layer_index]
        h_t = hs_by_layer[layer_index][t + 1]
        h_prev = hs_by_layer[layer_index][t]
        layer_input = (
            inputs[t] if layer_index == 0 else hs_by_layer[layer_index - 1][t + 1]
        )

        dh = gradient_from_above + dh_next_by_layer[layer_index]
        dh_raw = (1 - h_t ** 2) * dh  # tanh'(h) = 1 - h^2

        updated_layer_gradients[layer_index] = {
            "dbh": layer_grads[layer_index]["dbh"] + dh_raw,
            "dWxh": layer_grads[layer_index]["dWxh"] + dh_raw @ layer_input.T,
            "dWhh": layer_grads[layer_index]["dWhh"] + dh_raw @ h_prev.T,
        }

        updated_next_hidden_gradients[layer_index] = layer.Whh.T @ dh_raw
        gradient_from_above = layer.Wxh.T @ dh_raw  # hands off to the layer below
    return updated_layer_gradients, updated_next_hidden_gradients


def stacked_backpropagate_through_time(
    params: StackedRNNParams,
    inputs: Sequence[np.ndarray],
    targets: Sequence[int],
    hs_by_layer: Sequence[Sequence[np.ndarray]],
    ps: Sequence[np.ndarray],
) -> StackedGradients:
    """Backpropagate the cross-entropy loss through every layer and every
    time step. `hs_by_layer[layer]` uses forward_sequence's usual layout:
    index 0 is that layer's initial hidden state, index t + 1 is its state
    at time t.
    """
    dWhy = np.zeros_like(params.Why)
    dby = np.zeros_like(params.by)
    layer_grads, dh_next_by_layer = _initialize_layer_gradients(params, hs_by_layer)

    for t in reversed(range(len(inputs))):
        dy = ps[t].copy()
        dy[targets[t], 0] -= 1.0
        dWhy += dy @ hs_by_layer[-1][t + 1].T
        dby += dy

        gradient_from_above = params.Why.T @ dy  # flows into the top layer first
        layer_grads, dh_next_by_layer = _accumulate_layer_gradients(
            params, inputs, hs_by_layer, t, gradient_from_above,
            layer_grads, dh_next_by_layer,
        )

    grads = StackedGradients(
        layers=tuple(LayerGradients(**fields) for fields in layer_grads),
        dWhy=dWhy,
        dby=dby,
    )
    return grads
