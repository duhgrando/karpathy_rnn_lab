"""§ 'Getting fancy' -- the LSTM's gated update, in place of the vanilla
RNN's single tanh."""
import numpy as np

from domain.lstm_model import (
    compute_gates,
    initialize_lstm_parameters,
    lstm_forward_sequence,
    lstm_step,
    sigmoid,
)
from domain.vocabulary import build_vocabulary, encode, one_hot


def test_lstm_params_expose_their_own_hidden_and_vocab_size():
    vocab = build_vocabulary("helo")
    params = initialize_lstm_parameters(vocab.size, hidden_size=8)

    assert params.hidden_size == 8
    assert params.vocab_size == vocab.size


def test_gates_are_squashed_into_their_expected_ranges():
    """Input/forget/output gates are sigmoids (0, 1); the candidate update
    is a tanh (-1, 1) -- exactly like the vanilla RNN's hidden state."""
    vocab = build_vocabulary("helo")
    params = initialize_lstm_parameters(vocab.size, hidden_size=8, seed=1)
    h_prev = np.zeros((8, 1))
    x = one_hot(vocab, 0)

    i, f, o, g = compute_gates(params, x, h_prev)

    for gate in (i, f, o):
        assert np.all(gate > 0.0) and np.all(gate < 1.0)
    assert np.all(g >= -1.0) and np.all(g <= 1.0)


def test_forget_gate_starts_near_one_so_the_cell_defaults_to_remembering():
    """The forget-gate bias trick this project's initialize_lstm_parameters uses
    (Jozefowicz et al., 2015): with bf initialized to 1 and small random
    weights, sigmoid(~1) keeps most of the previous cell state around by
    default, rather than the network having to learn to stop forgetting
    from scratch."""
    vocab = build_vocabulary("helo")
    params = initialize_lstm_parameters(vocab.size, hidden_size=8, seed=1)
    h_prev = np.zeros((8, 1))
    x = one_hot(vocab, 0)

    _, f, _, _ = compute_gates(params, x, h_prev)

    assert np.all(f > 0.6)


def test_cell_state_blends_forget_and_input_contributions():
    """c_t = f*c_prev + i*g -- if the input gate is (near) zero, the cell
    state should barely move from c_prev regardless of the candidate g."""
    vocab = build_vocabulary("helo")
    params = initialize_lstm_parameters(vocab.size, hidden_size=4, seed=2)
    h_prev = np.zeros((4, 1))
    c_prev = np.array([[0.5], [-0.3], [0.1], [0.9]])
    x = one_hot(vocab, 0)

    h_t, c_t, _, _ = lstm_step(params, x, h_prev, c_prev)

    assert h_t.shape == (4, 1)
    assert c_t.shape == (4, 1)
    assert np.all(np.abs(h_t) <= 1.0)  # h = o * tanh(c) is tanh-bounded


def test_lstm_step_never_mutates_its_inputs():
    vocab = build_vocabulary("helo")
    params = initialize_lstm_parameters(vocab.size, hidden_size=6, seed=3)
    h_prev = np.zeros((6, 1))
    c_prev = np.zeros((6, 1))
    x = one_hot(vocab, 0)
    h_before, c_before, x_before = h_prev.copy(), c_prev.copy(), x.copy()

    lstm_step(params, x, h_prev, c_prev)

    assert np.array_equal(h_prev, h_before)
    assert np.array_equal(c_prev, c_before)
    assert np.array_equal(x, x_before)


def test_forward_sequence_keeps_initial_hidden_and_cell_states():
    vocab = build_vocabulary("helo")
    params = initialize_lstm_parameters(vocab.size, hidden_size=6, seed=4)
    inputs = [one_hot(vocab, i) for i in encode(vocab, "hell")]
    h0, c0 = np.zeros((6, 1)), np.zeros((6, 1))

    hs, cs, ys, ps = lstm_forward_sequence(params, inputs, h0, c0)

    assert np.array_equal(hs[0], h0)
    assert np.array_equal(cs[0], c0)


def test_forward_sequence_returns_one_state_per_input():
    vocab = build_vocabulary("helo")
    params = initialize_lstm_parameters(vocab.size, hidden_size=6, seed=4)
    inputs = [one_hot(vocab, i) for i in encode(vocab, "hell")]
    h0, c0 = np.zeros((6, 1)), np.zeros((6, 1))

    hs, cs, _, _ = lstm_forward_sequence(params, inputs, h0, c0)

    assert len(hs) == len(inputs) + 1
    assert len(cs) == len(inputs) + 1


def test_forward_sequence_returns_outputs_and_probabilities_per_input():
    vocab = build_vocabulary("helo")
    params = initialize_lstm_parameters(vocab.size, hidden_size=6, seed=4)
    inputs = [one_hot(vocab, i) for i in encode(vocab, "hell")]
    h0, c0 = np.zeros((6, 1)), np.zeros((6, 1))

    _, _, ys, ps = lstm_forward_sequence(params, inputs, h0, c0)

    assert len(ys) == len(ps) == len(inputs)
    assert all(np.isclose(p.sum(), 1.0) for p in ps)


def test_sigmoid_matches_its_closed_form_at_zero():
    assert np.isclose(sigmoid(np.zeros((1, 1)))[0, 0], 0.5)
