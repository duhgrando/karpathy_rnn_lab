import numpy as np
import pytest

from domain.stacked_lstm import (
    initialize_stacked_lstm_parameters,
    stacked_lstm_forward_sequence,
)
from domain.stacked_lstm_training import (
    stacked_lstm_backpropagate_through_time,
    stacked_lstm_cross_entropy_loss,
)
from domain.vocabulary import build_vocabulary, encode, one_hot

HIDDEN_SIZES = (3, 2)
LAYER_FIELDS = (
    "Wxi", "Whi", "bi", "Wxf", "Whf", "bf",
    "Wxo", "Who", "bo", "Wxg", "Whg", "bg",
)


def _loss(params, inputs, targets, h0s, c0s):
    _, _, _, probabilities, _ = stacked_lstm_forward_sequence(
        params, inputs, h0s, c0s
    )
    return stacked_lstm_cross_entropy_loss(probabilities, targets)


def _assert_parameter_gradient(params, matrix, analytic, inputs, targets, h0s, c0s):
    epsilon = 1e-5
    positions = np.random.default_rng(0).choice(
        matrix.size, size=min(3, matrix.size), replace=False
    )
    for position in positions:
        original = matrix.flat[position]
        matrix.flat[position] = original + epsilon
        loss_plus = _loss(params, inputs, targets, h0s, c0s)
        matrix.flat[position] = original - epsilon
        loss_minus = _loss(params, inputs, targets, h0s, c0s)
        matrix.flat[position] = original
        numerical = (loss_plus - loss_minus) / (2 * epsilon)
        actual = analytic.flat[position]
        relative_error = abs(numerical - actual) / max(
            1e-8, abs(numerical) + abs(actual)
        )
        assert relative_error < 1e-3 or abs(numerical - actual) < 1e-6


def _case():
    vocab = build_vocabulary("abc")
    params = initialize_stacked_lstm_parameters(vocab.size, HIDDEN_SIZES, seed=2)
    indices = encode(vocab, "abca")
    inputs = [one_hot(vocab, index) for index in indices[:-1]]
    targets = np.asarray(indices[1:], dtype=np.intp)[:, np.newaxis]
    h0s = tuple(np.zeros((size, 1)) for size in HIDDEN_SIZES)
    c0s = tuple(np.zeros((size, 1)) for size in HIDDEN_SIZES)
    hs, cs, _, probabilities, masks = stacked_lstm_forward_sequence(
        params, inputs, h0s, c0s
    )
    gradients = stacked_lstm_backpropagate_through_time(
        params, inputs, targets, hs, cs, probabilities, masks
    )
    return params, inputs, targets, h0s, c0s, gradients


@pytest.mark.parametrize("layer_index", [0, 1])
@pytest.mark.parametrize("field", LAYER_FIELDS)
def test_stacked_lstm_layer_gradients_match_numerical_gradients(layer_index, field):
    params, inputs, targets, h0s, c0s, gradients = _case()
    parameter = getattr(params.layers[layer_index], field)
    analytic = getattr(gradients.layers[layer_index], f"d{field}")

    _assert_parameter_gradient(params, parameter, analytic, inputs, targets, h0s, c0s)


@pytest.mark.parametrize("field", ["Why", "by"])
def test_stacked_lstm_output_gradients_match_numerical_gradients(field):
    params, inputs, targets, h0s, c0s, gradients = _case()

    _assert_parameter_gradient(
        params, getattr(params, field), getattr(gradients, f"d{field}"),
        inputs, targets, h0s, c0s,
    )