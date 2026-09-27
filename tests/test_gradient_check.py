"""§ 'we can run the backpropagation algorithm ... to figure out in what
direction we should adjust every one of its weights'.

This is a numerical gradient check: perturb each parameter matrix by a tiny
epsilon and compare the resulting change in loss to what `bptt` predicts
analytically. Karpathy's own reference gist (linked from the article) ships
exactly this kind of check as its way of proving BPTT is wired correctly --
this test is the direct executable form of that idea.

Note: this test *does* poke into a parameter matrix's array in place, via
`.flat[position] = ...`, then restores it before the assertion. That is a
deliberate, test-only escape hatch for finite-difference checking; the
domain functions themselves (bptt, forward_sequence, ...) never do this to
their arguments -- see test_forward_pass.py / test_adagrad_update.py, which
check that directly.
"""
import numpy as np
import pytest

from domain.rnn_model import cross_entropy_loss, forward_sequence, init_params
from domain.training import bptt
from domain.vocabulary import build_vocabulary, encode, one_hot


def _loss_for_params(params, inputs, targets, h0):
    _, _, ps = forward_sequence(params, inputs, h0)
    return cross_entropy_loss(ps, targets)


@pytest.mark.parametrize("field", ["Wxh", "Whh", "Why", "bh", "by"])
def test_analytic_gradient_matches_numerical_gradient(field):
    vocab = build_vocabulary("hello")
    params = init_params(vocab.size, hidden_size=6, seed=1)
    text_indices = encode(vocab, "hello")
    inputs = [one_hot(vocab, i) for i in text_indices[:-1]]
    targets = text_indices[1:]
    h0 = np.zeros((6, 1))

    hs, _, ps = forward_sequence(params, inputs, h0)
    grads, _ = bptt(params, inputs, targets, hs, ps)
    analytic = getattr(grads, f"d{field}")
    param_matrix = getattr(params, field)

    epsilon = 1e-4
    rng = np.random.default_rng(0)
    sample_positions = rng.choice(
        param_matrix.size, size=min(5, param_matrix.size), replace=False
    )

    for position in sample_positions:
        original_value = param_matrix.flat[position]

        param_matrix.flat[position] = original_value + epsilon
        loss_plus = _loss_for_params(params, inputs, targets, h0)

        param_matrix.flat[position] = original_value - epsilon
        loss_minus = _loss_for_params(params, inputs, targets, h0)

        param_matrix.flat[position] = original_value  # restore before asserting

        numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
        analytic_gradient = analytic.flat[position]
        relative_error = abs(numerical_gradient - analytic_gradient) / max(
            1e-8, abs(numerical_gradient) + abs(analytic_gradient)
        )

        assert relative_error < 1e-3


def test_dh0_matches_numerical_gradient_of_loss_with_respect_to_the_initial_state():
    """bptt's second return value, dh0 = dL/dh0, isn't used by ordinary
    training (truncated BPTT treats the carried-over hidden state as a
    constant) -- but tests/test_vanishing_gradient_comparison.py relies on
    it to measure gradient decay, so it needs its own numerical check
    before it's trusted for that."""
    vocab = build_vocabulary("hello")
    params = init_params(vocab.size, hidden_size=6, seed=1)
    text_indices = encode(vocab, "hello")
    inputs = [one_hot(vocab, i) for i in text_indices[:-1]]
    targets = text_indices[1:]
    h0 = np.zeros((6, 1))

    hs, _, ps = forward_sequence(params, inputs, h0)
    _, dh0 = bptt(params, inputs, targets, hs, ps)

    epsilon = 1e-4
    for row in range(h0.shape[0]):
        perturbed_plus = h0.copy()
        perturbed_plus[row, 0] += epsilon
        loss_plus = _loss_for_params(params, inputs, targets, perturbed_plus)

        perturbed_minus = h0.copy()
        perturbed_minus[row, 0] -= epsilon
        loss_minus = _loss_for_params(params, inputs, targets, perturbed_minus)

        numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
        relative_error = abs(numerical_gradient - dh0[row, 0]) / max(
            1e-8, abs(numerical_gradient) + abs(dh0[row, 0])
        )

        assert relative_error < 1e-3
