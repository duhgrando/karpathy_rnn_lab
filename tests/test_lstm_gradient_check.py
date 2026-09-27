"""Same technique as test_gradient_check.py, applied to the LSTM's gated
BPTT: perturb each of the 14 parameter matrices by a tiny epsilon and check
the resulting change in loss against lstm_bptt's analytic gradient. Given
how much more intricate the LSTM's chain rule is than the vanilla RNN's
(four gates, a separate cell-state path), this check is what makes
trusting domain/lstm_training.py's hand-derived math reasonable at all.
"""
import numpy as np
import pytest

from domain.lstm_model import init_lstm_params, lstm_forward_sequence
from domain.lstm_training import lstm_bptt
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
    params = init_lstm_params(vocab.size, hidden_size=6, seed=1)
    text_indices = encode(vocab, "hello")
    inputs = [one_hot(vocab, i) for i in text_indices[:-1]]
    targets = text_indices[1:]
    h0, c0 = np.zeros((6, 1)), np.zeros((6, 1))

    hs, cs, _, ps = lstm_forward_sequence(params, inputs, h0, c0)
    grads, _, _ = lstm_bptt(params, inputs, targets, hs, cs, ps)
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


def test_dh0_and_dc0_match_numerical_gradients_of_the_initial_state():
    """lstm_bptt's extra return values -- dh0 = dL/dh0 and dc0 = dL/dc0 --
    are what tests/test_vanishing_gradient_comparison.py measures gradient
    retention with, so they need the same numerical trust as the weight
    gradients above."""
    vocab = build_vocabulary("hello")
    params = init_lstm_params(vocab.size, hidden_size=6, seed=1)
    text_indices = encode(vocab, "hello")
    inputs = [one_hot(vocab, i) for i in text_indices[:-1]]
    targets = text_indices[1:]
    h0, c0 = np.zeros((6, 1)), np.zeros((6, 1))

    hs, cs, _, ps = lstm_forward_sequence(params, inputs, h0, c0)
    _, dh0, dc0 = lstm_bptt(params, inputs, targets, hs, cs, ps)

    epsilon = 1e-4
    for name, base_state, analytic in [("h0", h0, dh0), ("c0", c0, dc0)]:
        for row in range(base_state.shape[0]):
            plus, minus = base_state.copy(), base_state.copy()
            plus[row, 0] += epsilon
            minus[row, 0] -= epsilon

            if name == "h0":
                loss_plus = _loss_for_params(params, inputs, targets, plus, c0)
                loss_minus = _loss_for_params(params, inputs, targets, minus, c0)
            else:
                loss_plus = _loss_for_params(params, inputs, targets, h0, plus)
                loss_minus = _loss_for_params(params, inputs, targets, h0, minus)

            numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
            relative_error = abs(numerical_gradient - analytic[row, 0]) / max(
                1e-8, abs(numerical_gradient) + abs(analytic[row, 0])
            )

            assert relative_error < 1e-3
