"""Domain: stacking recurrent layers -- the article's "Going deep" section:

    y1 = rnn1.step(x)
    y = rnn2.step(y1)

Each layer is a plain vanilla-RNN hidden-state transition with no output
projection of its own (`LayerParams`: Wxh/Whh/bh only); only the *top* of
the stack gets projected to vocab-sized logits, via a single shared
`Why`/`by` living on `StackedRNNParams`. Layer 0's input is the one-hot
character vector; layer k>0's input is layer (k-1)'s hidden state at the
same time step -- exactly the `y1` feeding into `rnn2.step` above.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from domain.rnn_model import softmax


@dataclass(frozen=True)
class LayerParams:
    Wxh: np.ndarray  # (hidden_size, input_size) -- input_size is the layer
                     # below's hidden_size, or vocab_size for the bottom layer
    Whh: np.ndarray  # (hidden_size, hidden_size)
    bh: np.ndarray   # (hidden_size, 1)


@dataclass(frozen=True)
class StackedRNNParams:
    layers: Tuple[LayerParams, ...]
    Why: np.ndarray  # (vocab_size, top_layer_hidden_size)
    by: np.ndarray


def initialize_stacked_rnn_parameters(
    vocab_size: int, hidden_sizes: Sequence[int], seed: int = 0
) -> StackedRNNParams:
    """`hidden_sizes` has one entry per layer, bottom to top -- e.g.
    `(64, 64)` for a 2-layer stack with equal-sized layers."""
    rng = np.random.default_rng(seed)
    scale = 0.01
    layers: List[LayerParams] = []
    input_size = vocab_size
    for hidden_size in hidden_sizes:
        layers.append(
            LayerParams(
                Wxh=rng.standard_normal((hidden_size, input_size)) * scale,
                Whh=rng.standard_normal((hidden_size, hidden_size)) * scale,
                bh=np.zeros((hidden_size, 1)),
            )
        )
        input_size = hidden_size
    return StackedRNNParams(
        layers=tuple(layers),
        Why=rng.standard_normal((vocab_size, input_size)) * scale,
        by=np.zeros((vocab_size, 1)),
    )


def stacked_step(
    params: StackedRNNParams, x_t: np.ndarray, hs_prev: Sequence[np.ndarray]
):
    """One time step through every layer, bottom to top. Pure: returns new
    values, never mutates x_t or any entry of hs_prev."""
    layer_input = x_t
    new_hs = []
    for layer, h_prev in zip(params.layers, hs_prev):
        h_t = np.tanh(layer.Whh @ h_prev + layer.Wxh @ layer_input + layer.bh)
        new_hs.append(h_t)
        layer_input = h_t  # this layer's output is the next layer's input
    y_t = params.Why @ layer_input + params.by
    p_t = softmax(y_t)
    return tuple(new_hs), y_t, p_t


def _append_layer_states(
    hs_by_layer: Sequence[Sequence[np.ndarray]], current_hs: Sequence[np.ndarray]
) -> Tuple[Tuple[np.ndarray, ...], ...]:
    return tuple(
        tuple(layer_states) + (hidden_state,)
        for layer_states, hidden_state in zip(hs_by_layer, current_hs)
    )


def stacked_forward_sequence(
    params: StackedRNNParams, inputs: Sequence[np.ndarray], h0s: Sequence[np.ndarray]
):
    """Unroll stacked_step over a sequence of one-hot input vectors.

    Returns (hs_by_layer, ys, ps): hs_by_layer[layer] is a tuple of hidden
    states for that layer, with the same "one extra leading entry" layout
    forward_sequence() uses (hs_by_layer[layer][0] == h0s[layer]), so
    stacked_backpropagate_through_time never needs to re-derive it.
    """
    hs_by_layer = tuple((h0,) for h0 in h0s)
    ys, ps = [], []
    current_hs = tuple(h0s)
    for x_t in inputs:
        current_hs, y_t, p_t = stacked_step(params, x_t, current_hs)
        hs_by_layer = _append_layer_states(hs_by_layer, current_hs)
        ys.append(y_t)
        ps.append(p_t)
    return hs_by_layer, tuple(ys), tuple(ps)
