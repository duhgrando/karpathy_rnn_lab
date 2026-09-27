import numpy as np

from domain.stacked_lstm import (
    initialize_stacked_lstm_parameters,
    stacked_lstm_forward_sequence,
)
from domain.stacked_lstm_training import (
    stacked_lstm_backpropagate_through_time,
    stacked_lstm_cross_entropy_loss,
)
from domain.vocabulary import build_vocabulary, encode, one_hot


def _forward(params, inputs, h0s, c0s, training, seed):
    return stacked_lstm_forward_sequence(
        params, inputs, h0s, c0s,
        dropout_probability=0.5, training=training, seed=seed,
    )


def _assert_layer_histories_equal(left, right):
    for left_layer, right_layer in zip(left, right):
        for left_state, right_state in zip(left_layer, right_layer):
            assert np.array_equal(left_state, right_state)


def _assert_timestep_arrays_equal(left, right):
    for left_value, right_value in zip(left, right):
        assert np.array_equal(left_value, right_value)


def _assert_mask_has_dropped_units(masks):
    assert any(np.any(mask == 0.0) for mask in masks)


def _assert_masks_are_ones(masks):
    for mask in masks:
        assert np.array_equal(mask, np.ones_like(mask))


def test_dropout_is_seeded_and_inference_ignores_dropout_probability():
    vocab = build_vocabulary("abc")
    params = initialize_stacked_lstm_parameters(vocab.size, (4, 3), seed=8)
    indices = encode(vocab, "abca")
    inputs = [one_hot(vocab, index) for index in indices[:-1]]
    h0s = (np.zeros((4, 1)), np.zeros((3, 1)))
    c0s = (np.zeros((4, 1)), np.zeros((3, 1)))

    first = _forward(params, inputs, h0s, c0s, training=True, seed=11)
    repeated = _forward(params, inputs, h0s, c0s, training=True, seed=11)
    inference = _forward(params, inputs, h0s, c0s, training=False, seed=11)
    no_dropout = stacked_lstm_forward_sequence(params, inputs, h0s, c0s)

    _assert_layer_histories_equal(first[0], repeated[0])
    _assert_layer_histories_equal(first[1], repeated[1])
    _assert_timestep_arrays_equal(first[2], repeated[2])
    _assert_timestep_arrays_equal(first[3], repeated[3])
    _assert_timestep_arrays_equal(first[4][0], repeated[4][0])
    _assert_mask_has_dropped_units(first[4][0])
    _assert_layer_histories_equal(inference[0], no_dropout[0])
    _assert_layer_histories_equal(inference[1], no_dropout[1])
    _assert_timestep_arrays_equal(inference[2], no_dropout[2])
    _assert_timestep_arrays_equal(inference[3], no_dropout[3])
    _assert_masks_are_ones(inference[4][0])


def test_dropout_masks_are_used_by_backward_pass_for_finite_gradients():
    vocab = build_vocabulary("abc")
    params = initialize_stacked_lstm_parameters(vocab.size, (3, 2), seed=9)
    indices = encode(vocab, "abca")
    inputs = [one_hot(vocab, index) for index in indices[:-1]]
    targets = np.asarray(indices[1:], dtype=np.intp)[:, np.newaxis]
    h0s = (np.zeros((3, 1)), np.zeros((2, 1)))
    c0s = (np.zeros((3, 1)), np.zeros((2, 1)))
    hs, cs, _, probabilities, masks = stacked_lstm_forward_sequence(
        params, inputs, h0s, c0s, dropout_probability=0.5, training=True, seed=3
    )

    gradients = stacked_lstm_backpropagate_through_time(
        params, inputs, targets, hs, cs, probabilities, masks
    )
    loss = stacked_lstm_cross_entropy_loss(probabilities, targets)

    assert np.isfinite(loss)
    assert all(
        np.isfinite(getattr(layer, field)).all()
        for layer in gradients.layers
        for field in ("dWxi", "dWhi", "dWxf", "dWhf", "dWxo", "dWho", "dWxg", "dWhg")
    )