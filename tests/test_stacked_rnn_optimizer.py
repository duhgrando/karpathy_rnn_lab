"""Optimizer mechanics for the stack: clip_stacked_gradients and
stacked_adagrad_update walk every layer plus the shared output projection,
reusing domain/optimization.py's adagrad_step() per array."""
import numpy as np

from domain.stacked_rnn import init_stacked_params
from domain.stacked_training import (
    LayerGradients,
    StackedGradients,
    clip_stacked_gradients,
    stacked_adagrad_update,
    zero_stacked_memory,
)

HIDDEN_SIZES = (4, 3)


def _ones_gradients(params):
    layer_grads = tuple(
        LayerGradients(
            dWxh=np.ones_like(layer.Wxh),
            dWhh=np.ones_like(layer.Whh),
            dbh=np.ones_like(layer.bh),
        )
        for layer in params.layers
    )
    return StackedGradients(
        layers=layer_grads, dWhy=np.ones_like(params.Why), dby=np.ones_like(params.by)
    )


def test_zero_stacked_memory_matches_every_layers_shapes():
    params = init_stacked_params(vocab_size=4, hidden_sizes=HIDDEN_SIZES, seed=0)

    memory = zero_stacked_memory(params)

    for layer_memory, layer_params in zip(memory.layers, params.layers):
        assert layer_memory.mWxh.shape == layer_params.Wxh.shape
        assert np.all(layer_memory.mWxh == 0.0)
    assert memory.mWhy.shape == params.Why.shape


def test_clip_stacked_gradients_bounds_every_layer():
    params = init_stacked_params(vocab_size=4, hidden_sizes=HIDDEN_SIZES, seed=0)
    extreme = _ones_gradients(params)
    extreme.layers[0].dWxh[:] = 100.0

    clipped = clip_stacked_gradients(extreme, clip=5.0)

    assert np.all(clipped.layers[0].dWxh <= 5.0)


def test_stacked_adagrad_update_moves_every_layer_against_its_gradient():
    params = init_stacked_params(vocab_size=4, hidden_sizes=HIDDEN_SIZES, seed=1)
    memory = zero_stacked_memory(params)
    grads = _ones_gradients(params)

    new_params, new_memory = stacked_adagrad_update(params, grads, memory, learning_rate=0.5)

    for new_layer, old_layer in zip(new_params.layers, params.layers):
        assert np.all(new_layer.Wxh < old_layer.Wxh)
    assert np.all(new_params.Why < params.Why)


def test_stacked_adagrad_update_never_mutates_its_inputs():
    params = init_stacked_params(vocab_size=4, hidden_sizes=HIDDEN_SIZES, seed=2)
    memory = zero_stacked_memory(params)
    grads = _ones_gradients(params)
    original_first_layer_Wxh = params.layers[0].Wxh.copy()

    stacked_adagrad_update(params, grads, memory, learning_rate=0.5)

    assert np.array_equal(params.layers[0].Wxh, original_first_layer_Wxh)
