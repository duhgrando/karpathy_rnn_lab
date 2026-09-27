from dataclasses import fields

import numpy as np
import pytest

from application.stacked_lstm_train_service import (
    StackedLSTMTrainingConfig,
    _make_minibatches,
    _one_hot_batch,
    train_stacked_lstm,
)
from domain.stacked_lstm import (
    initialize_stacked_lstm_parameters,
    stacked_lstm_forward_sequence,
)
from domain.stacked_lstm_training import (
    stacked_lstm_backpropagate_through_time,
    stacked_lstm_cross_entropy_loss,
)
from domain.vocabulary import build_vocabulary, encode, one_hot

CORPUS = "abba" * 20
CONFIG = StackedLSTMTrainingConfig(
    hidden_sizes=(4, 3), seq_length=4, learning_rate=0.1, seed=3, batch_size=2
)


def _batch_case():
    vocab = build_vocabulary("abc")
    encoded = (encode(vocab, "abc"), encode(vocab, "bca"))
    inputs = (
        np.column_stack((one_hot(vocab, encoded[0][0]), one_hot(vocab, encoded[1][0]))),
        np.column_stack((one_hot(vocab, encoded[0][1]), one_hot(vocab, encoded[1][1]))),
    )
    targets = np.asarray(
        ((encoded[0][1], encoded[1][1]), (encoded[0][2], encoded[1][2])),
        dtype=np.intp,
    )
    params = initialize_stacked_lstm_parameters(vocab.size, (3, 2), seed=7)
    h0s = (np.zeros((3, 2)), np.zeros((2, 2)))
    c0s = (np.zeros((3, 2)), np.zeros((2, 2)))
    hs, cs, _, probabilities, masks = stacked_lstm_forward_sequence(
        params, inputs, h0s, c0s
    )
    gradients = stacked_lstm_backpropagate_through_time(
        params, inputs, targets, hs, cs, probabilities, masks
    )
    return params, inputs, targets, h0s, c0s, gradients


def _finite_difference(params, matrix, analytic, inputs, targets, h0s, c0s):
    epsilon = 1e-5
    for position in np.random.default_rng(2).choice(matrix.size, size=3, replace=False):
        original = matrix.flat[position]
        matrix.flat[position] = original + epsilon
        _, _, _, plus_probabilities, _ = stacked_lstm_forward_sequence(
            params, inputs, h0s, c0s
        )
        loss_plus = stacked_lstm_cross_entropy_loss(plus_probabilities, targets)
        matrix.flat[position] = original - epsilon
        _, _, _, minus_probabilities, _ = stacked_lstm_forward_sequence(
            params, inputs, h0s, c0s
        )
        loss_minus = stacked_lstm_cross_entropy_loss(minus_probabilities, targets)
        matrix.flat[position] = original
        numerical = (loss_plus - loss_minus) / (2 * epsilon)
        actual = analytic.flat[position]
        assert abs(numerical - actual) < 1e-6 or abs(numerical - actual) / max(
            1e-8, abs(numerical) + abs(actual)
        ) < 1e-3


def _parameter_arrays(params):
    arrays = [
        getattr(layer, field.name)
        for layer in params.layers
        for field in fields(layer)
    ]
    return arrays + [params.Why, params.by]


def _fixed_window_loss(params, vocab):
    indices = encode(vocab, CORPUS)
    input_ids, target_ids = next(
        _make_minibatches(indices, CONFIG.seq_length, CONFIG.batch_size)
    )
    inputs = tuple(_one_hot_batch(vocab, time_step) for time_step in input_ids)
    targets = np.asarray(target_ids, dtype=np.intp)
    h0s = tuple(np.zeros((size, CONFIG.batch_size)) for size in CONFIG.hidden_sizes)
    c0s = tuple(np.zeros_like(hidden) for hidden in h0s)
    _, _, _, probabilities, _ = stacked_lstm_forward_sequence(params, inputs, h0s, c0s)
    return stacked_lstm_cross_entropy_loss(probabilities, targets)


def _assert_finite_parameters(params):
    assert all(np.isfinite(array).all() for array in _parameter_arrays(params))


def _assert_snapshot_count(result, expected_count):
    assert len(result.snapshots) == expected_count


def _assert_parameter_sets_equal(left, right):
    assert all(
        np.array_equal(left_array, right_array)
        for left_array, right_array in zip(
            _parameter_arrays(left), _parameter_arrays(right)
        )
    )


def test_minibatches_keep_each_stream_contiguous_and_shift_targets():
    batches = tuple(_make_minibatches(tuple(range(12)), seq_length=2, batch_size=2))

    assert len(batches) == 2
    assert batches[0] == (((0, 6), (1, 7)), ((1, 7), (2, 8)))
    assert batches[1] == (((2, 8), (3, 9)), ((3, 9), (4, 10)))


def test_batched_gradients_match_finite_differences():
    params, inputs, targets, h0s, c0s, gradients = _batch_case()

    _finite_difference(
        params, params.layers[0].Wxi, gradients.layers[0].dWxi,
        inputs, targets, h0s, c0s,
    )
    _finite_difference(
        params, params.layers[1].Whf, gradients.layers[1].dWhf,
        inputs, targets, h0s, c0s,
    )
    _finite_difference(params, params.Why, gradients.dWhy, inputs, targets, h0s, c0s)


def test_minibatch_training_reduces_fixed_window_loss():
    vocab = build_vocabulary(CORPUS)
    initial = train_stacked_lstm(CORPUS, vocab, CONFIG, epochs=0)
    trained = train_stacked_lstm(CORPUS, vocab, CONFIG, epochs=8)

    assert _fixed_window_loss(trained.params, vocab) < _fixed_window_loss(initial.params, vocab)
    _assert_snapshot_count(trained, expected_count=8 * 9)
    _assert_finite_parameters(trained.params)


def test_dropout_training_is_reproducible_with_fixed_seed():
    vocab = build_vocabulary(CORPUS)
    dropout_config = StackedLSTMTrainingConfig(
        hidden_sizes=(4, 3), seq_length=4, seed=5, batch_size=2,
        dropout_probability=0.25,
    )

    first_dropout_run = train_stacked_lstm(CORPUS, vocab, dropout_config, epochs=1)
    second_dropout_run = train_stacked_lstm(CORPUS, vocab, dropout_config, epochs=1)

    _assert_parameter_sets_equal(first_dropout_run.params, second_dropout_run.params)


def test_zero_epoch_minibatch_training_returns_initial_parameters_and_no_snapshots():
    vocab = build_vocabulary(CORPUS)

    result = train_stacked_lstm(CORPUS, vocab, CONFIG, epochs=0)
    expected = initialize_stacked_lstm_parameters(vocab.size, CONFIG.hidden_sizes, CONFIG.seed)

    assert result.snapshots == ()
    assert all(
        np.array_equal(actual, wanted)
        for actual, wanted in zip(_parameter_arrays(result.params), _parameter_arrays(expected))
    )


@pytest.mark.parametrize(
    ("config", "epochs"),
    [
        (StackedLSTMTrainingConfig(batch_size=0), 1),
        (StackedLSTMTrainingConfig(dropout_probability=1.0), 0),
        (StackedLSTMTrainingConfig(seq_length=0), 1),
    ],
)
def test_invalid_training_configuration_fails_before_any_epochs(config, epochs):
    vocab = build_vocabulary(CORPUS)

    with pytest.raises(ValueError):
        train_stacked_lstm(CORPUS, vocab, config, epochs=epochs)