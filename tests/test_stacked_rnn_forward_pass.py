"""§ 'Going deep' -- "y1 = rnn1.step(x); y = rnn2.step(y1)": layer k's
hidden state becomes layer k+1's input, and only the top layer's hidden
state gets projected to vocab-sized output logits.
"""
import numpy as np
import pytest

from domain.stacked_rnn import initialize_stacked_rnn_parameters, stacked_forward_sequence, stacked_step
from domain.vocabulary import build_vocabulary, encode, one_hot


def test_layer_shapes_chain_correctly_bottom_to_top():
    """Layer 0 reads the vocab-sized one-hot input; layer 1 reads layer 0's
    hidden state; Why projects the *top* layer's hidden state to logits."""
    vocab = build_vocabulary("helo")
    params = initialize_stacked_rnn_parameters(vocab.size, hidden_sizes=(5, 3), seed=0)

    assert params.layers[0].Wxh.shape == (5, vocab.size)  # bottom layer <- one-hot input
    assert params.layers[1].Wxh.shape == (3, 5)  # top layer <- bottom layer's hidden state
    assert params.Why.shape == (vocab.size, 3)  # output <- top layer's hidden state


def test_stacked_step_produces_one_hidden_state_per_layer():
    vocab = build_vocabulary("helo")
    params = initialize_stacked_rnn_parameters(vocab.size, hidden_sizes=(5, 3), seed=1)
    h0s = [np.zeros((5, 1)), np.zeros((3, 1))]
    x = one_hot(vocab, 0)

    new_hs, y_t, p_t = stacked_step(params, x, h0s)

    assert len(new_hs) == 2
    assert new_hs[0].shape == (5, 1)
    assert new_hs[1].shape == (3, 1)


def test_stacked_step_produces_a_valid_probability_distribution():
    vocab = build_vocabulary("helo")
    params = initialize_stacked_rnn_parameters(vocab.size, hidden_sizes=(5, 3), seed=1)
    h0s = [np.zeros((5, 1)), np.zeros((3, 1))]
    x = one_hot(vocab, 0)

    _, _, p_t = stacked_step(params, x, h0s)

    assert p_t.shape == (vocab.size, 1)
    assert np.isclose(p_t.sum(), 1.0)


def test_a_deeper_stack_produces_a_different_output_than_a_shallow_one():
    """Not a claim about which is 'better' -- just that stacking genuinely
    changes what gets computed, rather than being a no-op wrapper."""
    vocab = build_vocabulary("helo")
    x = one_hot(vocab, 0)

    shallow = initialize_stacked_rnn_parameters(vocab.size, hidden_sizes=(5,), seed=2)
    deep = initialize_stacked_rnn_parameters(vocab.size, hidden_sizes=(5, 5), seed=2)

    _, y_shallow, _ = stacked_step(shallow, x, [np.zeros((5, 1))])
    _, y_deep, _ = stacked_step(deep, x, [np.zeros((5, 1)), np.zeros((5, 1))])

    assert not np.allclose(y_shallow, y_deep)


def test_stacked_step_never_mutates_its_inputs():
    vocab = build_vocabulary("helo")
    params = initialize_stacked_rnn_parameters(vocab.size, hidden_sizes=(4, 4), seed=3)
    h0s = [np.zeros((4, 1)), np.zeros((4, 1))]
    x = one_hot(vocab, 0)
    h0s_before = [h.copy() for h in h0s]
    x_before = x.copy()

    stacked_step(params, x, h0s)

    assert all(np.array_equal(h, h_before) for h, h_before in zip(h0s, h0s_before))
    assert np.array_equal(x, x_before)


def test_forward_sequence_keeps_initial_hidden_state_per_layer():
    vocab = build_vocabulary("helo")
    params = initialize_stacked_rnn_parameters(vocab.size, hidden_sizes=(5, 3), seed=4)
    inputs = [one_hot(vocab, i) for i in encode(vocab, "hell")]
    h0s = [np.zeros((5, 1)), np.zeros((3, 1))]

    hs_by_layer, ys, ps = stacked_forward_sequence(params, inputs, h0s)

    assert np.array_equal(hs_by_layer[0][0], h0s[0])
    assert np.array_equal(hs_by_layer[1][0], h0s[1])


@pytest.mark.parametrize("layer_index", (0, 1))
def test_forward_sequence_returns_one_state_per_input_and_layer(layer_index):
    vocab = build_vocabulary("helo")
    params = initialize_stacked_rnn_parameters(vocab.size, hidden_sizes=(5, 3), seed=4)
    inputs = [one_hot(vocab, i) for i in encode(vocab, "hell")]
    h0s = [np.zeros((5, 1)), np.zeros((3, 1))]

    hs_by_layer, _, _ = stacked_forward_sequence(params, inputs, h0s)

    assert len(hs_by_layer[layer_index]) == len(inputs) + 1


def test_forward_sequence_returns_one_output_and_probability_per_input():
    vocab = build_vocabulary("helo")
    params = initialize_stacked_rnn_parameters(vocab.size, hidden_sizes=(5, 3), seed=4)
    inputs = [one_hot(vocab, i) for i in encode(vocab, "hell")]
    h0s = [np.zeros((5, 1)), np.zeros((3, 1))]

    _, ys, ps = stacked_forward_sequence(params, inputs, h0s)

    assert len(ys) == len(ps) == len(inputs)
