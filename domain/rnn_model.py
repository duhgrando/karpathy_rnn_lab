"""Domain: the vanilla RNN forward pass and loss.

Directly implements the "RNN computation" section of the article:

    h_t = tanh(Whh @ h_{t-1} + Wxh @ x_t + bh)
    y_t = Why @ h_t + by
    p_t = softmax(y_t)

and the cross-entropy / Softmax classifier applied "on every output vector
simultaneously" that the article uses to train it.

Pure functional core: every function returns a new value and never mutates
an argument. RNNParams is a frozen dataclass; initialize_rnn_parameters is the only
"random" boundary and takes an explicit seed so it stays reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class RNNParams:
    Wxh: np.ndarray  # (hidden_size, vocab_size)
    Whh: np.ndarray  # (hidden_size, hidden_size)
    Why: np.ndarray  # (vocab_size, hidden_size)
    bh: np.ndarray   # (hidden_size, 1)
    by: np.ndarray   # (vocab_size, 1)

    @property
    def hidden_size(self) -> int:
        return self.Whh.shape[0]

    @property
    def vocab_size(self) -> int:
        return self.Why.shape[0]


def initialize_rnn_parameters(vocab_size: int, hidden_size: int, seed: int = 0) -> RNNParams:
    """Small random initialization -- the "random numbers" starting point
    the article describes training away from."""
    rng = np.random.default_rng(seed)
    scale = 0.01
    return RNNParams(
        Wxh=rng.standard_normal((hidden_size, vocab_size)) * scale,
        Whh=rng.standard_normal((hidden_size, hidden_size)) * scale,
        Why=rng.standard_normal((vocab_size, hidden_size)) * scale,
        bh=np.zeros((hidden_size, 1)),
        by=np.zeros((vocab_size, 1)),
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def forward_step(
    params: RNNParams, x_t: np.ndarray, h_prev: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One RNN time step. Pure: returns a new state, never mutates h_prev."""
    h_t = np.tanh(params.Whh @ h_prev + params.Wxh @ x_t + params.bh)
    y_t = params.Why @ h_t + params.by
    p_t = softmax(y_t)
    return h_t, y_t, p_t


def forward_sequence(
    params: RNNParams, inputs: Sequence[np.ndarray], h0: np.ndarray
):
    """Unroll forward_step over a sequence of one-hot input vectors.

    Returns (hs, ys, ps): hs has one extra leading entry, hs[0] == h0, so
    hs[t] is always the "h_{t-1}" that produced hs[t + 1] -- exactly what
    BPTT needs without recomputing the forward pass.
    """
    hs = [h0]
    ys = []
    ps = []
    h = h0
    for x_t in inputs:
        h, y, p = forward_step(params, x_t, h)
        hs.append(h)
        ys.append(y)
        ps.append(p)
    return tuple(hs), tuple(ys), tuple(ps)


def cross_entropy_loss(ps: Sequence[np.ndarray], target_indices: Sequence[int]) -> float:
    """Sum of -log p(correct char) over the sequence -- the Softmax /
    cross-entropy loss applied to every output vector simultaneously."""
    return float(sum(-np.log(p[t, 0] + 1e-12) for p, t in zip(ps, target_indices)))
