"""Application service: the training loop.

This is the one place in the project allowed to loop and to accept an
optional side-effecting callback (`on_snapshot`) -- everything it calls into
is a pure domain function. State is still never mutated in place: each step
produces a new, immutable TrainerState that gets threaded into the next one.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterator, Optional, Tuple

import numpy as np

from domain.rnn_model import (
    RNNParams,
    cross_entropy_loss,
    forward_sequence,
    init_params,
)
from domain.sampling import sample as sample_from_model  # re-exported for convenience
from domain.training import (
    AdagradMemory,
    adagrad_update,
    bptt,
    clip_gradients,
    zero_memory,
)
from domain.vocabulary import Vocabulary, encode, one_hot

__all__ = [
    "TrainingConfig",
    "TrainingSnapshot",
    "TrainerState",
    "train",
    "sample_from_model",
]


@dataclass(frozen=True)
class TrainingConfig:
    hidden_size: int = 100
    seq_length: int = 25
    learning_rate: float = 0.1
    seed: int = 0


@dataclass(frozen=True)
class TrainingSnapshot:
    """One reported checkpoint, handed to on_snapshot as training runs."""
    iteration: int
    smooth_loss: float
    params: RNNParams


@dataclass(frozen=True)
class TrainerState:
    params: RNNParams
    memory: AdagradMemory
    hidden: np.ndarray
    smooth_loss: float
    iteration: int


def _make_batches(
    indices: Tuple[int, ...], seq_length: int
) -> Iterator[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Chunk the corpus into (input, target) windows of seq_length, each
    target being the input shifted one character to the right."""
    last_start = len(indices) - seq_length - 1
    for start in range(0, max(last_start, 0), seq_length):
        yield indices[start:start + seq_length], indices[start + 1:start + seq_length + 1]


def _run_batch(
    vocab: Vocabulary,
    config: TrainingConfig,
    batch: Tuple[Tuple[int, ...], Tuple[int, ...]],
    state: TrainerState,
) -> TrainerState:
    input_ids, target_ids = batch
    inputs = [one_hot(vocab, i) for i in input_ids]

    hs, _, ps = forward_sequence(state.params, inputs, state.hidden)
    loss = cross_entropy_loss(ps, target_ids)
    grads = clip_gradients(bptt(state.params, inputs, target_ids, hs, ps))
    new_params, new_memory = adagrad_update(
        state.params, grads, state.memory, config.learning_rate
    )

    return TrainerState(
        params=new_params,
        memory=new_memory,
        hidden=hs[-1],
        smooth_loss=state.smooth_loss * 0.999 + loss * 0.001,
        iteration=state.iteration + 1,
    )


def _run_epoch(
    vocab: Vocabulary,
    config: TrainingConfig,
    indices: Tuple[int, ...],
    state: TrainerState,
    on_snapshot: Optional[Callable[[TrainingSnapshot], None]],
) -> TrainerState:
    state = replace(state, hidden=np.zeros_like(state.hidden))
    for batch in _make_batches(indices, config.seq_length):
        state = _run_batch(vocab, config, batch, state)
        if on_snapshot is not None:
            on_snapshot(TrainingSnapshot(state.iteration, state.smooth_loss, state.params))
    return state


def train(
    corpus: str,
    vocab: Vocabulary,
    config: TrainingConfig,
    epochs: int,
    on_snapshot: Optional[Callable[[TrainingSnapshot], None]] = None,
) -> RNNParams:
    """Run truncated BPTT training over `epochs` passes of the corpus.

    Mirrors Karpathy's reference training loop: chunk the corpus into
    fixed-length windows, carry the hidden state forward between
    consecutive windows within an epoch, and track an exponential moving
    average of the loss.
    """
    indices = encode(vocab, corpus)
    params = init_params(vocab.size, config.hidden_size, config.seed)
    state = TrainerState(
        params=params,
        memory=zero_memory(params),
        hidden=np.zeros((config.hidden_size, 1)),
        smooth_loss=-np.log(1.0 / vocab.size) * config.seq_length,
        iteration=0,
    )
    for _ in range(epochs):
        state = _run_epoch(vocab, config, indices, state, on_snapshot)
    return state.params
