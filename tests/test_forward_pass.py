"""§ 'RNN computation' -- h = tanh(Whh h + Wxh x + bh), y = Why h + by."""
import numpy as np

from domain.rnn_model import (
    RNNOutput,
    forward_sequence,
    forward_step,
    initialize_rnn_parameters,
)
from domain.vocabulary import build_vocabulary, encode, one_hot


def test_rnn_params_expose_their_own_hidden_and_vocab_size():
    vocab = build_vocabulary("helo")
    params = initialize_rnn_parameters(vocab.size, hidden_size=8)

    assert params.hidden_size == 8
    assert params.vocab_size == vocab.size


def test_forward_step_hidden_state_is_squashed_to_tanh_range():
    vocab = build_vocabulary("helo")
    params = initialize_rnn_parameters(vocab.size, hidden_size=8)
    h0 = np.zeros((8, 1))
    x = one_hot(vocab, 0)

    h1, _, _ = forward_step(params, x, h0)

    assert h1.shape == (8, 1)
    assert np.all(h1 >= -1.0) and np.all(h1 <= 1.0)


def test_forward_step_output_is_a_probability_distribution():
    vocab = build_vocabulary("helo")
    params = initialize_rnn_parameters(vocab.size, hidden_size=8)
    h0 = np.zeros((8, 1))
    x = one_hot(vocab, 0)

    y = forward_step(params, x, h0)
    _, _, p1 = y

    assert p1.shape == (vocab.size, 1)
    assert np.isclose(p1.sum(), 1.0)
    assert np.all(p1 >= 0)


def test_forward_step_returns_named_output_fields():
    vocab = build_vocabulary("helo")
    params = initialize_rnn_parameters(vocab.size, hidden_size=8)
    h0 = np.zeros((8, 1))
    x = one_hot(vocab, 0)

    y = forward_step(params, x, h0)

    assert (
        isinstance(y, RNNOutput),
        y.hidden_state.shape,
        y.logits.shape,
        y.probabilities.shape,
    ) == (True, (8, 1), (vocab.size, 1), (vocab.size, 1))


def test_forward_step_depends_on_hidden_state_not_only_current_input():
    """'the RNN therefore cannot rely on the input alone and must use its
    recurrent connection to keep track of the context'."""
    vocab = build_vocabulary("helo")
    params = initialize_rnn_parameters(vocab.size, hidden_size=8)
    x = one_hot(vocab, vocab.chars.index("l"))

    _, y_from_zero_state, _ = forward_step(params, x, np.zeros((8, 1)))
    _, y_from_nonzero_state, _ = forward_step(params, x, np.full((8, 1), 0.5))

    assert not np.allclose(y_from_zero_state, y_from_nonzero_state)


def test_forward_step_never_mutates_its_inputs():
    vocab = build_vocabulary("helo")
    params = initialize_rnn_parameters(vocab.size, hidden_size=8)
    h0 = np.zeros((8, 1))
    x = one_hot(vocab, 0)
    h0_before, x_before = h0.copy(), x.copy()

    forward_step(params, x, h0)

    assert np.array_equal(h0, h0_before)
    assert np.array_equal(x, x_before)


def test_forward_sequence_keeps_h0_as_the_initial_state():
    vocab = build_vocabulary("helo")
    params = initialize_rnn_parameters(vocab.size, hidden_size=8)
    inputs = [one_hot(vocab, i) for i in encode(vocab, "hell")]
    h0 = np.zeros((8, 1))

    hs, ys, ps = forward_sequence(params, inputs, h0)

    assert np.array_equal(hs[0], h0)


def test_forward_sequence_returns_one_state_per_input():
    vocab = build_vocabulary("helo")
    params = initialize_rnn_parameters(vocab.size, hidden_size=8)
    inputs = [one_hot(vocab, i) for i in encode(vocab, "hell")]
    h0 = np.zeros((8, 1))

    hs, _, _ = forward_sequence(params, inputs, h0)

    assert len(hs) == len(inputs) + 1


def test_forward_sequence_returns_one_output_and_probability_per_input():
    vocab = build_vocabulary("helo")
    params = initialize_rnn_parameters(vocab.size, hidden_size=8)
    inputs = [one_hot(vocab, i) for i in encode(vocab, "hell")]
    h0 = np.zeros((8, 1))

    _, ys, ps = forward_sequence(params, inputs, h0)

    assert len(ys) == len(inputs)
    assert len(ps) == len(inputs)
