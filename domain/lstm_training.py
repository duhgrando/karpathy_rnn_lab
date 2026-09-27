"""Domain: backpropagation through time for the LSTM cell.

Mirrors domain/training.py's backpropagate_through_time() exactly in shape (pure, accumulates
locally, returns a fresh Gradients-like value plus dh0/dc0), but the chain
rule genuinely differs from the vanilla RNN's because of the four gates and
the separate cell-state path, so it gets its own function rather than
trying to share backpropagate_through_time() via a flag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from domain.lstm_model import LSTMParams, compute_gates


@dataclass(frozen=True)
class LSTMGradients:
    dWxi: np.ndarray
    dWhi: np.ndarray
    dbi: np.ndarray
    dWxf: np.ndarray
    dWhf: np.ndarray
    dbf: np.ndarray
    dWxo: np.ndarray
    dWho: np.ndarray
    dbo: np.ndarray
    dWxg: np.ndarray
    dWhg: np.ndarray
    dbg: np.ndarray
    dWhy: np.ndarray
    dby: np.ndarray


@dataclass(frozen=True)
class LSTMMemory:
    mWxi: np.ndarray
    mWhi: np.ndarray
    mbi: np.ndarray
    mWxf: np.ndarray
    mWhf: np.ndarray
    mbf: np.ndarray
    mWxo: np.ndarray
    mWho: np.ndarray
    mbo: np.ndarray
    mWxg: np.ndarray
    mWhg: np.ndarray
    mbg: np.ndarray
    mWhy: np.ndarray
    mby: np.ndarray


def lstm_backpropagate_through_time(
    params: LSTMParams,
    inputs: Sequence[np.ndarray],
    targets: Sequence[int],
    hs: Sequence[np.ndarray],
    cs: Sequence[np.ndarray],
    ps: Sequence[np.ndarray],
) -> Tuple[LSTMGradients, np.ndarray, np.ndarray]:
    """Backpropagate the cross-entropy loss through the unrolled LSTM.

    hs[0]/cs[0] are h_{-1}/c_{-1}; hs[t + 1]/cs[t + 1] are the states
    produced at step t -- lstm_forward_sequence()'s layout.

    Returns (grads, dh0, dc0): dh0 = dL/dh0 and dc0 = dL/dc0, the same kind
    of "how far back did the gradient reach" signal backpropagate_through_time() returns for the
    vanilla RNN. See tests/test_vanishing_gradient_comparison.py.
    """
    dWxi, dWhi, dbi = np.zeros_like(params.Wxi), np.zeros_like(params.Whi), np.zeros_like(params.bi)
    dWxf, dWhf, dbf = np.zeros_like(params.Wxf), np.zeros_like(params.Whf), np.zeros_like(params.bf)
    dWxo, dWho, dbo = np.zeros_like(params.Wxo), np.zeros_like(params.Who), np.zeros_like(params.bo)
    dWxg, dWhg, dbg = np.zeros_like(params.Wxg), np.zeros_like(params.Whg), np.zeros_like(params.bg)
    dWhy, dby = np.zeros_like(params.Why), np.zeros_like(params.by)

    dh_next = np.zeros_like(hs[0])
    dc_next = np.zeros_like(cs[0])

    for t in reversed(range(len(inputs))):
        i, f, o, g = compute_gates(params, inputs[t], hs[t])
        c_prev, c_t, h_t = cs[t], cs[t + 1], hs[t + 1]
        tanh_c_t = np.tanh(c_t)

        dy = ps[t].copy()
        dy[targets[t], 0] -= 1.0
        dWhy += dy @ h_t.T
        dby += dy

        dh = params.Why.T @ dy + dh_next

        do_raw = (dh * tanh_c_t) * o * (1 - o)  # d(sigmoid)
        dc = dh * o * (1 - tanh_c_t ** 2) + dc_next  # d(tanh)
        di_raw = (dc * g) * i * (1 - i)
        dg_raw = (dc * i) * (1 - g ** 2)
        df_raw = (dc * c_prev) * f * (1 - f)
        dc_next = dc * f

        dWxi += di_raw @ inputs[t].T
        dWhi += di_raw @ hs[t].T
        dbi += di_raw
        dWxf += df_raw @ inputs[t].T
        dWhf += df_raw @ hs[t].T
        dbf += df_raw
        dWxo += do_raw @ inputs[t].T
        dWho += do_raw @ hs[t].T
        dbo += do_raw
        dWxg += dg_raw @ inputs[t].T
        dWhg += dg_raw @ hs[t].T
        dbg += dg_raw

        dh_next = (
            params.Whi.T @ di_raw
            + params.Whf.T @ df_raw
            + params.Who.T @ do_raw
            + params.Whg.T @ dg_raw
        )

    grads = LSTMGradients(
        dWxi=dWxi, dWhi=dWhi, dbi=dbi,
        dWxf=dWxf, dWhf=dWhf, dbf=dbf,
        dWxo=dWxo, dWho=dWho, dbo=dbo,
        dWxg=dWxg, dWhg=dWhg, dbg=dbg,
        dWhy=dWhy, dby=dby,
    )
    return grads, dh_next, dc_next
