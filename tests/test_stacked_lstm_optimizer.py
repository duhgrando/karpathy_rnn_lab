import numpy as np

from domain.stacked_lstm import initialize_stacked_lstm_parameters
from domain.stacked_lstm_training import (
    StackedLSTMLayerGradients,
    StackedLSTMGradients,
    clip_stacked_lstm_gradients,
    stacked_lstm_adagrad_update,
    zero_stacked_lstm_memory,
)

LAYER_PARAMETER_FIELDS = (
    "Wxi", "Whi", "bi", "Wxf", "Whf", "bf",
    "Wxo", "Who", "bo", "Wxg", "Whg", "bg",
)


def _unit_gradients(params, value=1.0):
    layers = tuple(
        StackedLSTMLayerGradients(
            **{f"d{name}": np.full_like(getattr(layer, name), value)
               for name in LAYER_PARAMETER_FIELDS}
        )
        for layer in params.layers
    )
    return StackedLSTMGradients(
        layers=layers,
        dWhy=np.full_like(params.Why, value),
        dby=np.full_like(params.by, value),
    )


def _assert_layer_memory_matches_params(parameter_layer, memory_layer):
    for name in LAYER_PARAMETER_FIELDS:
        accumulator = getattr(memory_layer, f"m{name}")
        assert accumulator.shape == getattr(parameter_layer, name).shape
        assert np.all(accumulator == 0.0)


def _assert_layer_memories_match_params(params, memory):
    for parameter_layer, memory_layer in zip(params.layers, memory.layers):
        _assert_layer_memory_matches_params(parameter_layer, memory_layer)


def test_stacked_lstm_memory_matches_all_layer_parameter_shapes():
    params = initialize_stacked_lstm_parameters(vocab_size=4, hidden_sizes=(3, 2), seed=0)

    memory = zero_stacked_lstm_memory(params)

    _assert_layer_memories_match_params(params, memory)
    assert memory.mWhy.shape == params.Why.shape
    assert memory.mby.shape == params.by.shape


def test_stacked_lstm_clipping_bounds_every_layer_and_output_gradient():
    params = initialize_stacked_lstm_parameters(vocab_size=4, hidden_sizes=(3, 2), seed=1)
    gradients = _unit_gradients(params, value=20.0)

    clipped = clip_stacked_lstm_gradients(gradients, clip=5.0)

    _assert_clipped_layer_gradients(clipped)
    assert np.all(np.abs(clipped.dWhy) <= 5.0)


def _assert_layer_gradient_clipped(layer):
    for field in (f"d{name}" for name in LAYER_PARAMETER_FIELDS):
        assert np.all(np.abs(getattr(layer, field)) <= 5.0)


def _assert_clipped_layer_gradients(gradients):
    for layer in gradients.layers:
        _assert_layer_gradient_clipped(layer)


def _assert_adagrad_updates_each_layer(updated, params):
    for new_layer, old_layer in zip(updated.layers, params.layers):
        assert np.all(new_layer.Wxi < old_layer.Wxi)


def _assert_original_layers_unchanged(original, params):
    assert all(
        np.array_equal(saved, layer.Wxi)
        for saved, layer in zip(original, params.layers)
    )


def test_stacked_lstm_adagrad_updates_all_layers_without_mutating_inputs():
    params = initialize_stacked_lstm_parameters(vocab_size=4, hidden_sizes=(3, 2), seed=2)
    memory = zero_stacked_lstm_memory(params)
    gradients = _unit_gradients(params)
    original = tuple(layer.Wxi.copy() for layer in params.layers)

    updated, updated_memory = stacked_lstm_adagrad_update(
        params, gradients, memory, learning_rate=0.5
    )

    _assert_adagrad_updates_each_layer(updated, params)
    assert np.all(updated.Why < params.Why)
    _assert_original_layers_unchanged(original, params)
    assert all(np.all(layer_memory.mWxi > 0.0) for layer_memory in updated_memory.layers)