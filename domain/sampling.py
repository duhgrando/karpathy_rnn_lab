"""Domain: autoregressive sampling.

Implements the article's "At test time" paragraph -- "we feed a character
into the RNN and get a distribution over what characters are likely to come
next. We sample from this distribution, and feed it right back in to get
the next letter" -- plus the "Temperature" section on sharpening or
flattening that distribution before sampling from it.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from domain.rnn_model import RNNParams, forward_step, softmax
from domain.vocabulary import Vocabulary, one_hot


def _temperature_scaled(p: np.ndarray, temperature: float) -> np.ndarray:
    """Re-derive the softmax at a different temperature: lower than 1 makes
    the distribution more confident/conservative, higher makes it more
    diverse (and more error-prone), exactly as the article describes."""
    if temperature == 1.0:
        return p
    logits = np.log(p + 1e-12) / temperature
    return softmax(logits)


def sample(
    params: RNNParams,
    vocab: Vocabulary,
    h0: np.ndarray,
    seed_index: int,
    length: int,
    temperature: float,
    seed: int,
) -> Tuple[int, ...]:
    """Generate `length` character indices, feeding each sampled character
    back in as the next input.

    A generator is created locally from `seed` rather than mutating a
    caller-owned random state, so sampling is reproducible for a given seed.
    """
    rng = np.random.default_rng(seed)
    h = h0
    x = one_hot(vocab, seed_index)
    generated = []
    for _ in range(length):
        h, _, p = forward_step(params, x, h)
        p_t = _temperature_scaled(p, temperature)
        next_index = int(rng.choice(vocab.size, p=p_t.ravel()))
        generated.append(next_index)
        x = one_hot(vocab, next_index)
    return tuple(generated)
