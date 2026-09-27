import numpy as np

from domain.stacked_lstm import (
    initialize_stacked_lstm_parameters,
    stacked_lstm_forward_sequence,
)
from domain.stacked_lstm_training import stacked_lstm_cross_entropy_loss
from domain.vocabulary import build_vocabulary, encode, one_hot


def _single_sequence_loss(params, vocab, indices):
    inputs = (one_hot(vocab, indices[0]), one_hot(vocab, indices[1]))
    targets = np.asarray((indices[1], indices[2]), dtype=np.intp)[:, np.newaxis]
    h0s = (np.zeros((3, 1)), np.zeros((2, 1)))
    c0s = (np.zeros((3, 1)), np.zeros((2, 1)))
    _, _, _, probabilities, _ = stacked_lstm_forward_sequence(
        params, inputs, h0s, c0s
    )
    return stacked_lstm_cross_entropy_loss(probabilities, targets)


def _batch_case():
    vocab = build_vocabulary("abc")
    params = initialize_stacked_lstm_parameters(vocab.size, (3, 2), seed=4)
    encoded = (encode(vocab, "abc"), encode(vocab, "cab"))
    inputs = (
        np.column_stack((one_hot(vocab, encoded[0][0]), one_hot(vocab, encoded[1][0]))),
        np.column_stack((one_hot(vocab, encoded[0][1]), one_hot(vocab, encoded[1][1]))),
    )
    targets = np.asarray(
        ((encoded[0][1], encoded[1][1]), (encoded[0][2], encoded[1][2])),
        dtype=np.intp,
    )
    h0s = (np.zeros((3, 2)), np.zeros((2, 2)))
    c0s = (np.zeros((3, 2)), np.zeros((2, 2)))
    _, _, _, batch_probabilities, _ = stacked_lstm_forward_sequence(
        params, inputs, h0s, c0s
    )
    return vocab, params, encoded, batch_probabilities, targets


def _assert_probabilities_normalize(probabilities):
    for probability in probabilities:
        assert np.allclose(probability.sum(axis=0), 1.0)


def test_batch_probabilities_normalize_and_batch_loss_is_mean_token_loss():
    vocab, params, encoded, batch_probabilities, targets = _batch_case()
    batch_loss = stacked_lstm_cross_entropy_loss(batch_probabilities, targets)
    individual_losses = tuple(
        _single_sequence_loss(params, vocab, indices) for indices in encoded
    )

    _assert_probabilities_normalize(batch_probabilities)
    assert np.isclose(batch_loss, np.mean(individual_losses))
    assert np.isclose(batch_loss, np.mean(individual_losses))


def test_single_column_batch_loss_matches_single_example_path():
    vocab = build_vocabulary("abc")
    params = initialize_stacked_lstm_parameters(vocab.size, (3, 2), seed=6)
    indices = encode(vocab, "abc")
    inputs = (one_hot(vocab, indices[0]), one_hot(vocab, indices[1]))
    targets = np.asarray((indices[1], indices[2]), dtype=np.intp)[:, np.newaxis]
    h0s = (np.zeros((3, 1)), np.zeros((2, 1)))
    c0s = (np.zeros((3, 1)), np.zeros((2, 1)))
    _, _, _, probabilities, _ = stacked_lstm_forward_sequence(
        params, inputs, h0s, c0s
    )

    batch_loss = stacked_lstm_cross_entropy_loss(probabilities, targets)
    single_loss = _single_sequence_loss(params, vocab, indices)

    assert np.isclose(batch_loss, single_loss)