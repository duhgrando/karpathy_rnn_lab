"""Domain: truncated backpropagation through time (BPTT) for the vanilla
RNN, matching Karpathy's reference min-char-rnn implementation linked from
the article ("a minimal character-level RNN language model in
Python/numpy").

backpropagate_through_time() is pure at its boundary: given the same arguments it returns the
same new values, and it never mutates params, hs, or ps. It accumulates
into local arrays for performance, exactly the way a hand-written BPTT loop
would -- but that bookkeeping is invisible to callers, who only ever see a
fresh Gradients value coming back.

Gradient clipping and the Adagrad update are cell-agnostic and live in
domain/optimization.py; import them from there.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from domain.rnn_model import RNNParams


@dataclass(frozen=True)
class Gradients:
    dWxh: np.ndarray
    dWhh: np.ndarray
    dWhy: np.ndarray
    dbh: np.ndarray
    dby: np.ndarray


@dataclass(frozen=True)
class AdagradMemory:
    mWxh: np.ndarray
    mWhh: np.ndarray
    mWhy: np.ndarray
    mbh: np.ndarray
    mby: np.ndarray


def backpropagate_through_time(
    params: RNNParams,
    inputs: Sequence[np.ndarray],
    targets: Sequence[int],
    hs: Sequence[np.ndarray],
    ps: Sequence[np.ndarray],
) -> Tuple[Gradients, np.ndarray]:
    """Backpropagate the cross-entropy loss through the unrolled sequence.

    hs[0] is h_{-1} (the state fed in before the first step); hs[t + 1] is
    the hidden state produced at step t -- the layout forward_sequence()
    returns, so callers never need to re-derive it.

    Returns (grads, dh0): `dh0` is dL/dh0, the gradient of the summed loss
    with respect to the *initial* hidden state. Ordinary training ignores
    it (truncated BPTT deliberately treats the carried-over hidden state as
    a constant, not something to backprop into) -- it exists so a caller
    can measure how much gradient a loss at the *end* of a long sequence
    still manages to push all the way back to the *start* of it. See
    tests/test_vanishing_gradient_comparison.py.
    """
    dWxh = np.zeros_like(params.Wxh)
    dWhh = np.zeros_like(params.Whh)
    dWhy = np.zeros_like(params.Why)
    dbh = np.zeros_like(params.bh)
    dby = np.zeros_like(params.by)
    dh_next = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = ps[t].copy()
        dy[targets[t], 0] -= 1.0  # d(cross_entropy)/d(logits) = softmax - one_hot(target)

        dWhy += dy @ hs[t + 1].T
        dby += dy

        dh = params.Why.T @ dy + dh_next
        dh_raw = (1 - hs[t + 1] ** 2) * dh  # tanh'(h) = 1 - h^2

        dbh += dh_raw
        dWxh += dh_raw @ inputs[t].T
        dWhh += dh_raw @ hs[t].T

        dh_next = params.Whh.T @ dh_raw

    grads = Gradients(dWxh=dWxh, dWhh=dWhh, dWhy=dWhy, dbh=dbh, dby=dby)
    dh0 = dh_next
    return grads, dh0
