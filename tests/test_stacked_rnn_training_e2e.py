import numpy as np

from application.stacked_train_service import (
    StackedTrainingConfig,
    train_stacked_rnn,
)
from domain.rnn_model import cross_entropy_loss
from domain.stacked_rnn import initialize_stacked_rnn_parameters, stacked_forward_sequence
from domain.vocabulary import build_vocabulary, encode, one_hot

CORPUS = "abba" * 8
CONFIG = StackedTrainingConfig(
    hidden_sizes=(4, 3), seq_length=4, learning_rate=0.1, seed=3
)


def _parameter_arrays(params):
    arrays = [array for layer in params.layers for array in (layer.Wxh, layer.Whh, layer.bh)]
    return arrays + [params.Why, params.by]


def _assert_finite_parameters(params):
    assert all(np.isfinite(array).all() for array in _parameter_arrays(params))


def _assert_batch_snapshots(result, expected_count):
    assert len(result.snapshots) == expected_count
    assert tuple(snapshot.iteration for snapshot in result.snapshots) == tuple(
        range(1, expected_count + 1)
    )


def _loss(params, vocab):
    indices = encode(vocab, CORPUS)
    inputs = [one_hot(vocab, index) for index in indices[:CONFIG.seq_length]]
    targets = indices[1:CONFIG.seq_length + 1]
    initial = tuple(np.zeros((size, 1)) for size in CONFIG.hidden_sizes)
    _, _, probabilities = stacked_forward_sequence(params, inputs, initial)
    return cross_entropy_loss(probabilities, targets)


def test_stacked_training_reduces_fixed_window_loss_and_records_snapshots():
    vocab = build_vocabulary(CORPUS)

    initial = train_stacked_rnn(CORPUS, vocab, CONFIG, epochs=0)
    trained = train_stacked_rnn(CORPUS, vocab, CONFIG, epochs=8)

    assert _loss(trained.params, vocab) < _loss(initial.params, vocab)
    _assert_finite_parameters(trained.params)
    _assert_batch_snapshots(trained, expected_count=8 * 7)


def test_zero_epoch_stack_training_returns_seeded_initial_parameters():
    vocab = build_vocabulary(CORPUS)

    result = train_stacked_rnn(CORPUS, vocab, CONFIG, epochs=0)
    expected = initialize_stacked_rnn_parameters(
        vocab.size, CONFIG.hidden_sizes, CONFIG.seed
    )

    assert result.snapshots == ()
    assert all(
        np.array_equal(actual, wanted)
        for actual, wanted in zip(_parameter_arrays(result.params), _parameter_arrays(expected))
    )


def test_stack_snapshots_do_not_change_after_later_updates():
    vocab = build_vocabulary(CORPUS)

    result = train_stacked_rnn(CORPUS, vocab, CONFIG, epochs=2)
    first_snapshot = _parameter_arrays(result.snapshots[0].params)
    final_params = _parameter_arrays(result.params)

    assert all(not np.shares_memory(before, after) for before, after in zip(first_snapshot, final_params))
    assert any(not np.array_equal(before, after) for before, after in zip(first_snapshot, final_params))