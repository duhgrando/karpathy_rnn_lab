"""Domain: truncated backpropagation through time (BPTT) and the Adagrad
parameter update, matching Karpathy's reference min-char-rnn implementation
linked from the article ("a minimal character-level RNN language model in
Python/numpy").

Every function is pure at its boundary: given the same arguments it returns
the same new value, and it never mutates params, hs, or ps. bptt() and the
per-parameter `step` closure inside adagrad_update() accumulate into local
arrays for performance, exactly the way a hand-written BPTT loop would --
but that bookkeeping is invisible to callers, who only ever see fresh
Gradients / RNNParams / AdagradMemory values coming back.
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


def bptt(
    params: RNNParams,
    inputs: Sequence[np.ndarray],
    targets: Sequence[int],
    hs: Sequence[np.ndarray],
    ps: Sequence[np.ndarray],
) -> Gradients:
    """Backpropagate the cross-entropy loss through the unrolled sequence.

    hs[0] is h_{-1} (the state fed in before the first step); hs[t + 1] is
    the hidden state produced at step t -- the layout forward_sequence()
    returns, so callers never need to re-derive it.
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

    return Gradients(dWxh=dWxh, dWhh=dWhh, dWhy=dWhy, dbh=dbh, dby=dby)


def clip_gradients(grads: Gradients, clip: float = 5.0) -> Gradients:
    """Mitigate exploding gradients, as the reference implementation does."""
    return Gradients(
        dWxh=np.clip(grads.dWxh, -clip, clip),
        dWhh=np.clip(grads.dWhh, -clip, clip),
        dWhy=np.clip(grads.dWhy, -clip, clip),
        dbh=np.clip(grads.dbh, -clip, clip),
        dby=np.clip(grads.dby, -clip, clip),
    )


@dataclass(frozen=True)
class AdagradMemory:
    mWxh: np.ndarray
    mWhh: np.ndarray
    mWhy: np.ndarray
    mbh: np.ndarray
    mby: np.ndarray


def zero_memory(params: RNNParams) -> AdagradMemory:
    return AdagradMemory(
        mWxh=np.zeros_like(params.Wxh),
        mWhh=np.zeros_like(params.Whh),
        mWhy=np.zeros_like(params.Why),
        mbh=np.zeros_like(params.bh),
        mby=np.zeros_like(params.by),
    )


def _adagrad_step(
    param: np.ndarray, grad: np.ndarray, mem: np.ndarray, learning_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    eps = 1e-8
    new_mem = mem + grad * grad
    new_param = param - learning_rate * grad / np.sqrt(new_mem + eps)
    return new_param, new_mem


def adagrad_update(
    params: RNNParams,
    grads: Gradients,
    memory: AdagradMemory,
    learning_rate: float = 0.1,
) -> Tuple[RNNParams, AdagradMemory]:
    """Per-parameter adaptive learning rate update."""
    Wxh, mWxh = _adagrad_step(params.Wxh, grads.dWxh, memory.mWxh, learning_rate)
    Whh, mWhh = _adagrad_step(params.Whh, grads.dWhh, memory.mWhh, learning_rate)
    Why, mWhy = _adagrad_step(params.Why, grads.dWhy, memory.mWhy, learning_rate)
    bh, mbh = _adagrad_step(params.bh, grads.dbh, memory.mbh, learning_rate)
    by, mby = _adagrad_step(params.by, grads.dby, memory.mby, learning_rate)

    new_params = RNNParams(Wxh=Wxh, Whh=Whh, Why=Why, bh=bh, by=by)
    new_memory = AdagradMemory(mWxh=mWxh, mWhh=mWhh, mWhy=mWhy, mbh=mbh, mby=mby)
    return new_params, new_memory
