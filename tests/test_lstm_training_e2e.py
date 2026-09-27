import dataclasses

import numpy as np

from application.lstm_train_service import train_lstm
from application.train_service import TrainingConfig
from domain.lstm_model import initialize_lstm_parameters, lstm_forward_sequence
from domain.rnn_model import cross_entropy_loss
from domain.vocabulary import build_vocabulary, encode, one_hot

CORPUS = "abba" * 8
CONFIG = TrainingConfig(hidden_size=8, seq_length=4, learning_rate=0.1, seed=3)


def _loss(params, vocab):
    indices = encode(vocab, CORPUS)
    inputs = [one_hot(vocab, index) for index in indices[:CONFIG.seq_length]]
    targets = indices[1:CONFIG.seq_length + 1]
    initial = np.zeros((CONFIG.hidden_size, 1))
    _, _, _, probabilities = lstm_forward_sequence(
        params, inputs, initial, initial.copy()
    )
    return cross_entropy_loss(probabilities, targets)


def _parameter_arrays(params):
    return [getattr(params, field.name) for field in dataclasses.fields(params)]


def _assert_finite_parameters(params):
    assert all(np.isfinite(array).all() for array in _parameter_arrays(params))


def _assert_batch_snapshots(result, expected_count):
    assert len(result.snapshots) == expected_count
    assert tuple(snapshot.iteration for snapshot in result.snapshots) == tuple(
        range(1, expected_count + 1)
    )


def test_lstm_training_reduces_fixed_window_loss_and_records_batch_snapshots():
    vocab = build_vocabulary(CORPUS)

    initial = train_lstm(CORPUS, vocab, CONFIG, epochs=0)
    trained = train_lstm(CORPUS, vocab, CONFIG, epochs=8)

    assert _loss(trained.params, vocab) < _loss(initial.params, vocab)
    _assert_finite_parameters(trained.params)
    _assert_batch_snapshots(trained, expected_count=8 * 7)


def test_zero_epoch_lstm_training_returns_seeded_initial_parameters():
    vocab = build_vocabulary(CORPUS)

    result = train_lstm(CORPUS, vocab, CONFIG, epochs=0)
    expected = initialize_lstm_parameters(vocab.size, CONFIG.hidden_size, CONFIG.seed)

    assert result.snapshots == ()
    assert all(
        np.array_equal(actual, getattr(expected, field.name))
        for field, actual in zip(dataclasses.fields(expected), _parameter_arrays(result.params))
    )


def test_lstm_snapshots_do_not_change_after_later_updates():
    vocab = build_vocabulary(CORPUS)

    result = train_lstm(CORPUS, vocab, CONFIG, epochs=2)
    first_snapshot = _parameter_arrays(result.snapshots[0].params)
    final_params = _parameter_arrays(result.params)

    assert all(not np.shares_memory(before, after) for before, after in zip(first_snapshot, final_params))
    assert any(not np.array_equal(before, after) for before, after in zip(first_snapshot, final_params))