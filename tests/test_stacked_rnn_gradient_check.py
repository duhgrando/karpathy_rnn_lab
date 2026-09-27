"""Same technique as test_gradient_check.py, applied across every layer of
a stacked RNN: perturb each parameter matrix by a tiny epsilon and check
the resulting change in loss against stacked_bptt's analytic gradient.
This is what makes stacked_bptt's "hand gradients down through the stack,
layer by layer" derivation (see domain/stacked_training.py's docstring)
trustworthy rather than merely plausible.

Uses a smaller epsilon (1e-5) and a combined relative-or-absolute
tolerance: some gradients here (particularly `by`, on a 5-character toy
corpus) are genuinely tiny, and central-difference approximation error
starts to dominate the *relative* error long before the analytic gradient
is actually wrong -- an absolute floor avoids that false positive.
"""
import numpy as np
import pytest

from domain.rnn_model import cross_entropy_loss
from domain.stacked_rnn import init_stacked_params, stacked_forward_sequence
from domain.stacked_training import stacked_bptt
from domain.vocabulary import build_vocabulary, encode, one_hot

HIDDEN_SIZES = (5, 4)


def _loss_for_params(params, inputs, targets, h0s):
    _, _, ps = stacked_forward_sequence(params, inputs, h0s)
    return cross_entropy_loss(ps, targets)


def _assert_matches_numerical_gradient(param_matrix, analytic_grad, params, inputs, targets, h0s):
    epsilon = 1e-5
    rng = np.random.default_rng(0)
    positions = rng.choice(param_matrix.size, size=min(4, param_matrix.size), replace=False)

    for position in positions:
        original_value = param_matrix.flat[position]

        param_matrix.flat[position] = original_value + epsilon
        loss_plus = _loss_for_params(params, inputs, targets, h0s)

        param_matrix.flat[position] = original_value - epsilon
        loss_minus = _loss_for_params(params, inputs, targets, h0s)

        param_matrix.flat[position] = original_value  # restore before asserting

        numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
        analytic_gradient = analytic_grad.flat[position]
        relative_error = abs(numerical_gradient - analytic_gradient) / max(
            1e-8, abs(numerical_gradient) + abs(analytic_gradient)
        )
        absolute_error = abs(numerical_gradient - analytic_gradient)

        assert relative_error < 1e-3 or absolute_error < 1e-6


@pytest.mark.parametrize("layer_index", [0, 1])
@pytest.mark.parametrize("field", ["Wxh", "Whh", "bh"])
def test_layer_gradients_match_numerical_gradients(layer_index, field):
    vocab = build_vocabulary("hello")
    params = init_stacked_params(vocab.size, HIDDEN_SIZES, seed=1)
    text_indices = encode(vocab, "hello")
    inputs = [one_hot(vocab, i) for i in text_indices[:-1]]
    targets = text_indices[1:]
    h0s = [np.zeros((size, 1)) for size in HIDDEN_SIZES]

    hs_by_layer, _, ps = stacked_forward_sequence(params, inputs, h0s)
    grads = stacked_bptt(params, inputs, targets, hs_by_layer, ps)

    param_matrix = getattr(params.layers[layer_index], field)
    analytic_grad = getattr(grads.layers[layer_index], f"d{field}")
    _assert_matches_numerical_gradient(param_matrix, analytic_grad, params, inputs, targets, h0s)


@pytest.mark.parametrize("field", ["Why", "by"])
def test_output_projection_gradients_match_numerical_gradients(field):
    vocab = build_vocabulary("hello")
    params = init_stacked_params(vocab.size, HIDDEN_SIZES, seed=1)
    text_indices = encode(vocab, "hello")
    inputs = [one_hot(vocab, i) for i in text_indices[:-1]]
    targets = text_indices[1:]
    h0s = [np.zeros((size, 1)) for size in HIDDEN_SIZES]

    hs_by_layer, _, ps = stacked_forward_sequence(params, inputs, h0s)
    grads = stacked_bptt(params, inputs, targets, hs_by_layer, ps)

    param_matrix = getattr(params, field)
    analytic_grad = getattr(grads, f"d{field}")
    _assert_matches_numerical_gradient(param_matrix, analytic_grad, params, inputs, targets, h0s)
