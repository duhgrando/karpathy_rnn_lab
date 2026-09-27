import numpy as np
import pytest

from domain.stacked_lstm import (
    initialize_stacked_lstm_parameters,
    stacked_lstm_forward_sequence,
)
from domain.vocabulary import build_vocabulary, encode, one_hot


def _assert_state_histories_are_finite(histories):
    for layer_states in histories:
        for state in layer_states:
            assert np.isfinite(state).all()


def _assert_mask_collection_shape(masks, input_count):
    assert len(masks) == 1
    assert len(masks[0]) == input_count


def _assert_default_mask_values(masks, hidden_size):
    for mask in masks[0]:
        assert np.array_equal(mask, np.ones((hidden_size, 1)))


def _assert_layer_columns_match(batch_layers, single_layers, column):
    for batch_layer, single_layer in zip(batch_layers, single_layers):
        for batch_state, single_state in zip(batch_layer, single_layer):
            assert np.allclose(batch_state[:, [column]], single_state)


def _assert_probability_columns_match(batch_probabilities, single_probabilities, column):
    for batch_probabilities_t, single_probabilities_t in zip(
        batch_probabilities, single_probabilities
    ):
        assert np.allclose(batch_probabilities_t[:, [column]], single_probabilities_t)


def _assert_history_lengths(hs, cs):
    assert tuple(map(len, hs)) == (3, 3)
    assert tuple(map(len, cs)) == (3, 3)


def _assert_hidden_shapes(hs):
    assert hs[0][1].shape == (3, 1)
    assert hs[1][1].shape == (2, 1)


def _assert_probabilities_normalize(probabilities):
    assert all(np.allclose(probability.sum(axis=0), 1.0) for probability in probabilities)


@pytest.mark.parametrize("hidden_sizes", [(), (0,), (3, -1)])
def test_initializer_rejects_empty_or_nonpositive_hidden_sizes(hidden_sizes):
    with pytest.raises(ValueError):
        initialize_stacked_lstm_parameters(vocab_size=3, hidden_sizes=hidden_sizes)


def test_stacked_lstm_forward_chains_layer_shapes_and_normalizes_outputs():
    vocab = build_vocabulary("abc")
    params = initialize_stacked_lstm_parameters(vocab.size, (3, 2), seed=4)
    indices = encode(vocab, "ab")
    inputs = [one_hot(vocab, index) for index in indices]
    h0s = (np.zeros((3, 1)), np.zeros((2, 1)))
    c0s = (np.zeros((3, 1)), np.zeros((2, 1)))

    hs, cs, logits, probabilities, masks = stacked_lstm_forward_sequence(
        params, inputs, h0s, c0s
    )

    _assert_history_lengths(hs, cs)
    _assert_hidden_shapes(hs)
    assert logits[0].shape == (vocab.size, 1)
    _assert_probabilities_normalize(probabilities)
    _assert_state_histories_are_finite(hs + cs)
    _assert_mask_collection_shape(masks, len(inputs))
    _assert_default_mask_values(masks, hidden_size=3)


def test_forward_rejects_wrong_number_of_initial_states():
    params = initialize_stacked_lstm_parameters(vocab_size=3, hidden_sizes=(3, 2), seed=1)
    h0s = (np.zeros((3, 1)),)
    c0s = (np.zeros((3, 1)),)

    with pytest.raises(ValueError):
        stacked_lstm_forward_sequence(params, (), h0s, c0s)


def test_forward_rejects_dropout_probability_outside_unit_interval():
    params = initialize_stacked_lstm_parameters(vocab_size=3, hidden_sizes=(3, 2), seed=1)
    h0s = (np.zeros((3, 1)), np.zeros((2, 1)))
    c0s = (np.zeros((3, 1)), np.zeros((2, 1)))

    with pytest.raises(ValueError):
        stacked_lstm_forward_sequence(
            params, (), h0s, c0s, dropout_probability=1.0, training=True
        )


def test_batched_forward_matches_independent_single_example_forwards():
    vocab = build_vocabulary("abc")
    params = initialize_stacked_lstm_parameters(vocab.size, (3, 2), seed=5)
    sequences = ("ab", "ca")
    encoded = [encode(vocab, sequence) for sequence in sequences]
    batched_inputs = [
        np.column_stack([one_hot(vocab, encoded[column][time]) for column in range(2)])
        for time in range(2)
    ]
    batched_h0s = (np.zeros((3, 2)), np.zeros((2, 2)))
    batched_c0s = (np.zeros((3, 2)), np.zeros((2, 2)))

    batch_hs, batch_cs, _, batch_ps, _ = stacked_lstm_forward_sequence(
        params, batched_inputs, batched_h0s, batched_c0s
    )

    _assert_batch_matches_single_sequences(
        params, vocab, encoded, batch_hs, batch_cs, batch_ps
    )

    _assert_probabilities_normalize(batch_ps)


def _assert_batch_matches_single_sequences(params, vocab, encoded, batch_hs, batch_cs, batch_ps):
    for column, sequence_indices in enumerate(encoded):
        inputs = [one_hot(vocab, index) for index in sequence_indices]
        h0s = (np.zeros((3, 1)), np.zeros((2, 1)))
        c0s = (np.zeros((3, 1)), np.zeros((2, 1)))
        hs, cs, _, ps, _ = stacked_lstm_forward_sequence(params, inputs, h0s, c0s)
        _assert_layer_columns_match(batch_hs, hs, column)
        _assert_layer_columns_match(batch_cs, cs, column)
        _assert_probability_columns_match(batch_ps, ps, column)