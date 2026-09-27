"""Same technique as test_gradient_check.py, applied to the LSTM's gated
BPTT: perturb each of the 14 parameter matrices by a tiny epsilon and check
the resulting change in loss against lstm_backpropagate_through_time's analytic gradient. Given
how much more intricate the LSTM's chain rule is than the vanilla RNN's
(four gates, a separate cell-state path), this check is what makes
trusting domain/lstm_training.py's hand-derived math reasonable at all.
"""
import numpy as np
import pytest

from domain.lstm_model import initialize_lstm_parameters, lstm_forward_sequence
from domain.lstm_training import lstm_backpropagate_through_time
from domain.rnn_model import cross_entropy_loss
from domain.vocabulary import build_vocabulary, encode, one_hot

LSTM_PARAM_FIELDS = [
    "Wxi", "Whi", "bi",
    "Wxf", "Whf", "bf",
    "Wxo", "Who", "bo",
    "Wxg", "Whg", "bg",
    "Why", "by",
]


def _loss_for_params(params, inputs, targets, h0, c0):
    _, _, _, ps = lstm_forward_sequence(params, inputs, h0, c0)
    return cross_entropy_loss(ps, targets)


@pytest.mark.parametrize("field", LSTM_PARAM_FIELDS)
def test_analytic_gradient_matches_numerical_gradient(field):
    vocab = build_vocabulary("hello")
    params = initialize_lstm_parameters(vocab.size, hidden_size=6, seed=1)
    text_indices = encode(vocab, "hello")
    inputs = [one_hot(vocab, i) for i in text_indices[:-1]]
    targets = text_indices[1:]
    h0, c0 = np.zeros((6, 1)), np.zeros((6, 1))

    hs, cs, _, ps = lstm_forward_sequence(params, inputs, h0, c0)
    grads, _, _ = lstm_backpropagate_through_time(params, inputs, targets, hs, cs, ps)
    analytic = getattr(grads, f"d{field}")
    param_matrix = getattr(params, field)

    epsilon = 1e-4
    rng = np.random.default_rng(0)
    sample_positions = rng.choice(
        param_matrix.size, size=min(4, param_matrix.size), replace=False
    )

    for position in sample_positions:
        original_value = param_matrix.flat[position]

        param_matrix.flat[position] = original_value + epsilon
        loss_plus = _loss_for_params(params, inputs, targets, h0, c0)

        param_matrix.flat[position] = original_value - epsilon
        loss_minus = _loss_for_params(params, inputs, targets, h0, c0)

        param_matrix.flat[position] = original_value  # restore before asserting

        numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
        analytic_gradient = analytic.flat[position]
        relative_error = abs(numerical_gradient - analytic_gradient) / max(
            1e-8, abs(numerical_gradient) + abs(analytic_gradient)
        )

        assert relative_error < 1e-3


def _assert_initial_state_gradient_matches_numerical_gradient(
    initial_state, analytic_gradient, loss_for_state
):
    epsilon = 1e-4
    for row in range(initial_state.shape[0]):
        plus, minus = initial_state.copy(), initial_state.copy()
        plus[row, 0] += epsilon
        minus[row, 0] -= epsilon

        loss_plus = loss_for_state(plus)
        loss_minus = loss_for_state(minus)
        numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
        relative_error = abs(numerical_gradient - analytic_gradient[row, 0]) / max(
            1e-8, abs(numerical_gradient) + abs(analytic_gradient[row, 0])
        )

        assert relative_error < 1e-3


def _initial_state_gradient_case():
    vocab = build_vocabulary("hello")
    params = initialize_lstm_parameters(vocab.size, hidden_size=6, seed=1)
    text_indices = encode(vocab, "hello")
    inputs = [one_hot(vocab, i) for i in text_indices[:-1]]
    targets = text_indices[1:]
    h0, c0 = np.zeros((6, 1)), np.zeros((6, 1))

    hs, cs, _, ps = lstm_forward_sequence(params, inputs, h0, c0)
    _, dh0, dc0 = lstm_backpropagate_through_time(params, inputs, targets, hs, cs, ps)
    return params, inputs, targets, h0, c0, dh0, dc0


def test_initial_hidden_state_gradient_matches_numerical_gradient():
    """Check dh0, which the vanishing-gradient comparison uses."""
    params, inputs, targets, h0, c0, dh0, _ = _initial_state_gradient_case()
    loss_for_hidden = lambda state: _loss_for_params(params, inputs, targets, state, c0)

    _assert_initial_state_gradient_matches_numerical_gradient(h0, dh0, loss_for_hidden)


def test_initial_cell_state_gradient_matches_numerical_gradient():
    """Check dc0, which the vanishing-gradient comparison uses."""
    params, inputs, targets, h0, c0, _, dc0 = _initial_state_gradient_case()
    loss_for_cell = lambda state: _loss_for_params(params, inputs, targets, h0, state)

    _assert_initial_state_gradient_matches_numerical_gradient(c0, dc0, loss_for_cell)
