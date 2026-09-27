"""Domain: autoregressive sampling.

Implements the article's "At test time" paragraph -- "we feed a character
into the RNN and get a distribution over what characters are likely to come
next. We sample from this distribution, and feed it right back in to get
the next letter" -- plus the "Temperature" section on sharpening or
flattening that distribution before sampling from it.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple, TypeVar

import numpy as np

from domain.lstm_model import LSTMParams, lstm_step
from domain.rnn_model import RNNParams, forward_step, softmax
from domain.stacked_rnn import StackedRNNParams, stacked_step
from domain.stacked_lstm import StackedLSTMParams, stacked_lstm_step
from domain.vocabulary import Vocabulary, one_hot

State = TypeVar("State")


def _temperature_scaled(p: np.ndarray, temperature: float) -> np.ndarray:
    """Re-derive the softmax at a different temperature: lower than 1 makes
    the distribution more confident/conservative, higher makes it more
    diverse (and more error-prone), exactly as the article describes."""
    if temperature == 1.0:
        return p
    logits = np.log(p + 1e-12) / temperature
    return softmax(logits)


def _sample(
    vocab: Vocabulary,
    initial_state: State,
    seed_index: int,
    length: int,
    temperature: float,
    seed: int,
    advance: Callable[[State, np.ndarray], Tuple[State, np.ndarray]],
) -> Tuple[int, ...]:
    rng = np.random.default_rng(seed)
    state = initial_state
    x = one_hot(vocab, seed_index)
    generated = []
    for _ in range(length):
        state, p = advance(state, x)
        p_t = _temperature_scaled(p, temperature)
        next_index = int(rng.choice(vocab.size, p=p_t.ravel()))
        generated.append(next_index)
        x = one_hot(vocab, next_index)
    return tuple(generated)


def _advance_rnn(params: RNNParams, state: np.ndarray, x: np.ndarray):
    hidden, _, probabilities = forward_step(params, x, state)
    return hidden, probabilities


def sample(
    params: RNNParams,
    vocab: Vocabulary,
    h0: np.ndarray,
    seed_index: int,
    length: int,
    temperature: float,
    seed: int,
) -> Tuple[int, ...]:
    """Generate characters autoregressively from a vanilla RNN."""
    return _sample(
        vocab, h0, seed_index, length, temperature, seed,
        lambda state, x: _advance_rnn(params, state, x),
    )


def _advance_lstm(params: LSTMParams, state, x: np.ndarray):
    hidden, cell = state
    next_hidden, next_cell, _, probabilities = lstm_step(params, x, hidden, cell)
    return (next_hidden, next_cell), probabilities


def sample_lstm(
    params: LSTMParams,
    vocab: Vocabulary,
    h0: np.ndarray,
    c0: np.ndarray,
    seed_index: int,
    length: int,
    temperature: float,
    seed: int,
) -> Tuple[int, ...]:
    """Generate characters autoregressively from an LSTM."""
    return _sample(
        vocab, (h0, c0), seed_index, length, temperature, seed,
        lambda state, x: _advance_lstm(params, state, x),
    )


def _advance_stacked_rnn(
    params: StackedRNNParams, state: Sequence[np.ndarray], x: np.ndarray
):
    next_hidden, _, probabilities = stacked_step(params, x, state)
    return next_hidden, probabilities


def sample_stacked_rnn(
    params: StackedRNNParams,
    vocab: Vocabulary,
    h0s: Sequence[np.ndarray],
    seed_index: int,
    length: int,
    temperature: float,
    seed: int,
) -> Tuple[int, ...]:
    """Generate characters autoregressively from a stack of vanilla RNN layers."""
    return _sample(
        vocab, tuple(h0s), seed_index, length, temperature, seed,
        lambda state, x: _advance_stacked_rnn(params, state, x),
    )


def _advance_stacked_lstm(params: StackedLSTMParams, state, x: np.ndarray):
    hidden, cell = state
    next_hidden, next_cell, _, probabilities = stacked_lstm_step(
        params, x, hidden, cell
    )
    return (next_hidden, next_cell), probabilities


def sample_stacked_lstm(
    params: StackedLSTMParams,
    vocab: Vocabulary,
    h0s: Sequence[np.ndarray],
    c0s: Sequence[np.ndarray],
    seed_index: int,
    length: int,
    temperature: float,
    seed: int,
) -> Tuple[int, ...]:
    """Generate characters autoregressively from stacked LSTM layers."""
    return _sample(
        vocab, (tuple(h0s), tuple(c0s)), seed_index, length, temperature, seed,
        lambda state, x: _advance_stacked_lstm(params, state, x),
    )
