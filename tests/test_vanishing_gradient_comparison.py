"""§ 'Getting fancy' -- "the LSTM ... works slightly better in practice,
owing to its more powerful update equation and some appealing
backpropagation dynamics."

This test measures that "appealing backpropagation dynamics" claim
directly, without training anything: seed a gradient at the *last* time
step of a 20-step sequence (as if some future loss depended only on
getting that last character right), backpropagate purely through the
recurrence with no local per-step loss in the way at any earlier step, and
compare how much of that gradient survives all the way back to the *first*
time step. That's exactly what backpropagate_through_time()'s/lstm_backpropagate_through_time()'s `dh0` return value
is -- see test_gradient_check.py for its own numerical check, which this
test's conclusions depend on.

The vanilla RNN's gradient decays by roughly 20+ orders of magnitude over
these 20 steps; the LSTM's decays by only a handful. That gap -- not
whether either model has been *trained* on anything -- is the actual
mechanism the article is pointing at, and it shows up in a single forward
+ backward pass through freshly-initialized parameters.
"""
import numpy as np

from domain.lstm_model import initialize_lstm_parameters, lstm_forward_sequence
from domain.lstm_training import lstm_backpropagate_through_time
from domain.rnn_model import forward_sequence, initialize_rnn_parameters
from domain.training import backpropagate_through_time as rnn_backpropagate_through_time
from domain.vocabulary import build_vocabulary, one_hot

SEQUENCE_LENGTH = 20
HIDDEN_SIZE = 16


def _one_hot_vector(index: int, size: int) -> np.ndarray:
    vector = np.zeros((size, 1))
    vector[index, 0] = 1.0
    return vector


def _sequence_with_loss_only_at_the_last_step(vocab, rng):
    """Build (inputs, targets, ps) where `ps` is rigged to be a *perfect*
    prediction at every step except the last -- so the per-step output
    gradient (softmax - one_hot(target)) that backpropagate_through_time/lstm_backpropagate_through_time compute is
    exactly zero everywhere except at the final position. Only a loss on
    the very last character puts any gradient into this backward pass."""
    input_indices = rng.integers(0, vocab.size, size=SEQUENCE_LENGTH)
    inputs = [one_hot(vocab, i) for i in input_indices]
    targets = list(input_indices)

    perfect_ps = [_one_hot_vector(t, vocab.size) for t in targets]
    imperfect_last_prediction = np.full((vocab.size, 1), 1.0 / vocab.size)
    ps = perfect_ps[:-1] + [imperfect_last_prediction]
    return inputs, targets, ps


def test_lstm_carries_gradient_back_through_far_more_steps_than_vanilla_rnn():
    vocab = build_vocabulary("abcd")
    rng = np.random.default_rng(7)
    inputs, targets, ps = _sequence_with_loss_only_at_the_last_step(vocab, rng)
    h0 = np.zeros((HIDDEN_SIZE, 1))
    c0 = np.zeros((HIDDEN_SIZE, 1))

    rnn_params = initialize_rnn_parameters(vocab.size, HIDDEN_SIZE, seed=3)
    rnn_hs, _, _ = forward_sequence(rnn_params, inputs, h0)
    _, rnn_dh0 = rnn_backpropagate_through_time(rnn_params, inputs, targets, rnn_hs, ps)

    lstm_params = initialize_lstm_parameters(vocab.size, HIDDEN_SIZE, seed=3)
    lstm_hs, lstm_cs, _, _ = lstm_forward_sequence(lstm_params, inputs, h0, c0)
    _, lstm_dh0, _ = lstm_backpropagate_through_time(lstm_params, inputs, targets, lstm_hs, lstm_cs, ps)

    rnn_gradient_at_step_0 = np.linalg.norm(rnn_dh0)
    lstm_gradient_at_step_0 = np.linalg.norm(lstm_dh0)

    # Observed magnitudes across many seeds: RNN ~1e-29 to 1e-31, LSTM
    # ~1e-6 to 1e-7 -- a gap of roughly 20+ orders of magnitude. The
    # thresholds below leave an enormous safety margin rather than pinning
    # an exact ratio.
    assert rnn_gradient_at_step_0 < 1e-15, "expected the vanilla RNN's gradient to have vanished"
    assert lstm_gradient_at_step_0 > 1e-8, "expected the LSTM's gradient to still be present"
    assert lstm_gradient_at_step_0 > rnn_gradient_at_step_0 * 1e6
