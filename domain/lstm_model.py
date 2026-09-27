"""Domain: the LSTM cell -- the article's "Getting fancy" section, where a
"more complex update equation" (gated, with a separate cell state) replaces
the vanilla RNN's single tanh:

    i_t = sigmoid(Wxi x_t + Whi h_{t-1} + bi)   # input gate
    f_t = sigmoid(Wxf x_t + Whf h_{t-1} + bf)   # forget gate
    o_t = sigmoid(Wxo x_t + Who h_{t-1} + bo)   # output gate
    g_t = tanh   (Wxg x_t + Whg h_{t-1} + bg)   # candidate update
    c_t = f_t * c_{t-1} + i_t * g_t             # new cell state
    h_t = o_t * tanh(c_t)                       # new hidden state
    y_t = Why h_t + by                          # same output projection as the vanilla RNN

Same pure-functional shape as domain/rnn_model.py: LSTMParams is a frozen
dataclass, initialize_lstm_parameters is the only random boundary, and every function
returns new values rather than mutating its arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from domain.rnn_model import softmax


@dataclass(frozen=True)
class LSTMParams:
    Wxi: np.ndarray
    Whi: np.ndarray
    bi: np.ndarray
    Wxf: np.ndarray
    Whf: np.ndarray
    bf: np.ndarray
    Wxo: np.ndarray
    Who: np.ndarray
    bo: np.ndarray
    Wxg: np.ndarray
    Whg: np.ndarray
    bg: np.ndarray
    Why: np.ndarray
    by: np.ndarray

    @property
    def hidden_size(self) -> int:
        return self.Whi.shape[0]

    @property
    def vocab_size(self) -> int:
        return self.Why.shape[0]


def initialize_lstm_parameters(vocab_size: int, hidden_size: int, seed: int = 0) -> LSTMParams:
    """Small random initialization, matching initialize_rnn_parameters(). The forget gate
    bias starts at 1 rather than 0 -- a well-known trick (Jozefowicz et al.,
    2015) that makes the cell default to *remembering* early in training,
    which is exactly what tests/test_vanishing_gradient_comparison.py relies
    on to show the LSTM out-retaining the vanilla RNN."""
    rng = np.random.default_rng(seed)
    scale = 0.01

    def weight(rows, cols):
        return rng.standard_normal((rows, cols)) * scale

    return LSTMParams(
        Wxi=weight(hidden_size, vocab_size), Whi=weight(hidden_size, hidden_size), bi=np.zeros((hidden_size, 1)),
        Wxf=weight(hidden_size, vocab_size), Whf=weight(hidden_size, hidden_size), bf=np.ones((hidden_size, 1)),
        Wxo=weight(hidden_size, vocab_size), Who=weight(hidden_size, hidden_size), bo=np.zeros((hidden_size, 1)),
        Wxg=weight(hidden_size, vocab_size), Whg=weight(hidden_size, hidden_size), bg=np.zeros((hidden_size, 1)),
        Why=weight(vocab_size, hidden_size), by=np.zeros((vocab_size, 1)),
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_gates(
    params: LSTMParams, x_t: np.ndarray, h_prev: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The four gate activations for one step. Factored out of lstm_step so
    lstm_backpropagate_through_time can recompute the *exact* same values during backprop instead
    of caching a parallel copy of them."""
    i = sigmoid(params.Wxi @ x_t + params.Whi @ h_prev + params.bi)
    f = sigmoid(params.Wxf @ x_t + params.Whf @ h_prev + params.bf)
    o = sigmoid(params.Wxo @ x_t + params.Who @ h_prev + params.bo)
    g = np.tanh(params.Wxg @ x_t + params.Whg @ h_prev + params.bg)
    return i, f, o, g


def lstm_step(
    params: LSTMParams, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
):
    """One LSTM time step. Pure: returns new (h, c, y, p), never mutates
    h_prev or c_prev."""
    i, f, o, g = compute_gates(params, x_t, h_prev)
    c_t = f * c_prev + i * g
    h_t = o * np.tanh(c_t)
    y_t = params.Why @ h_t + params.by
    p_t = softmax(y_t)
    return h_t, c_t, y_t, p_t


def lstm_forward_sequence(
    params: LSTMParams, inputs: Sequence[np.ndarray], h0: np.ndarray, c0: np.ndarray
):
    """Unroll lstm_step over a sequence of one-hot input vectors.

    Returns (hs, cs, ys, ps): hs/cs each have one extra leading entry
    (hs[0] == h0, cs[0] == c0), the same layout forward_sequence() uses for
    the vanilla RNN, so lstm_backpropagate_through_time never needs to re-derive it.
    """
    hs, cs, ys, ps = [h0], [c0], [], []
    h, c = h0, c0
    for x_t in inputs:
        h, c, y, p = lstm_step(params, x_t, h, c)
        hs.append(h)
        cs.append(c)
        ys.append(y)
        ps.append(p)
    return tuple(hs), tuple(cs), tuple(ys), tuple(ps)
