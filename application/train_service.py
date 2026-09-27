"""Application service: the training loop.

This service loops over epochs and batches, but returns its outputs as data
and never invokes callbacks or mutates caller-owned state. Each step produces
a new TrainerState that gets threaded into the next one.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np

from domain.rnn_model import (
    RNNParams,
    cross_entropy_loss,
    forward_sequence,
    initialize_rnn_parameters,
)
from domain.optimization import adagrad_update, clip_gradients, zero_memory
from domain.sampling import sample as sample_from_model  # re-exported for convenience
from domain.training import AdagradMemory, backpropagate_through_time
from domain.vocabulary import Vocabulary, encode, one_hot
from application.training_utils import make_character_batches as _make_batches

__all__ = [
    "TrainingConfig",
    "TrainingSnapshot",
    "TrainingResult",
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
    """One checkpoint captured after a training batch."""
    iteration: int
    smooth_loss: float
    params: RNNParams


@dataclass(frozen=True)
class TrainingResult:
    """Final parameters and immutable per-batch training snapshots."""
    params: RNNParams
    snapshots: Tuple[TrainingSnapshot, ...]


@dataclass(frozen=True)
class TrainerState:
    params: RNNParams
    memory: AdagradMemory
    hidden: np.ndarray
    smooth_loss: float
    iteration: int


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
    grads, _ = backpropagate_through_time(state.params, inputs, target_ids, hs, ps)
    grads = clip_gradients(grads)
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
) -> Tuple[TrainerState, Tuple[TrainingSnapshot, ...]]:
    state = replace(state, hidden=np.zeros_like(state.hidden))
    snapshots = ()
    for batch in _make_batches(indices, config.seq_length):
        state = _run_batch(vocab, config, batch, state)
        snapshots += (TrainingSnapshot(state.iteration, state.smooth_loss, state.params),)
    return state, snapshots


def train(
    corpus: str,
    vocab: Vocabulary,
    config: TrainingConfig,
    epochs: int,
) -> TrainingResult:
    """Run truncated BPTT training over `epochs` passes of the corpus.

    Mirrors Karpathy's reference training loop: chunk the corpus into
    fixed-length windows, carry the hidden state forward between
    consecutive windows within an epoch, and track an exponential moving
    average of the loss. Return the final parameters and each batch snapshot
    as data, without invoking a callback.
    """
    indices = encode(vocab, corpus)
    params = initialize_rnn_parameters(vocab.size, config.hidden_size, config.seed)
    state = TrainerState(
        params=params,
        memory=zero_memory(params, AdagradMemory),
        hidden=np.zeros((config.hidden_size, 1)),
        smooth_loss=-np.log(1.0 / vocab.size) * config.seq_length,
        iteration=0,
    )
    snapshots = ()
    for _ in range(epochs):
        state, epoch_snapshots = _run_epoch(vocab, config, indices, state)
        snapshots += epoch_snapshots
    return TrainingResult(params=state.params, snapshots=snapshots)
