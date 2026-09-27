"""Application service for training a stack of vanilla RNN layers."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np

from application.training_utils import make_character_batches
from domain.rnn_model import cross_entropy_loss
from domain.stacked_rnn import (
    StackedRNNParams,
    initialize_stacked_rnn_parameters,
    stacked_forward_sequence,
)
from domain.stacked_training import (
    StackedMemory,
    clip_stacked_gradients,
    stacked_adagrad_update,
    stacked_backpropagate_through_time,
    zero_stacked_memory,
)
from domain.vocabulary import Vocabulary, encode, one_hot

__all__ = [
    "StackedTrainingConfig",
    "StackedTrainingSnapshot",
    "StackedTrainingResult",
    "StackedTrainerState",
    "train_stacked_rnn",
]


@dataclass(frozen=True)
class StackedTrainingConfig:
    hidden_sizes: Tuple[int, ...] = (100, 100)
    seq_length: int = 25
    learning_rate: float = 0.1
    seed: int = 0


@dataclass(frozen=True)
class StackedTrainingSnapshot:
    iteration: int
    smooth_loss: float
    params: StackedRNNParams


@dataclass(frozen=True)
class StackedTrainingResult:
    params: StackedRNNParams
    snapshots: Tuple[StackedTrainingSnapshot, ...]


@dataclass(frozen=True)
class StackedTrainerState:
    params: StackedRNNParams
    memory: StackedMemory
    hidden: Tuple[np.ndarray, ...]
    smooth_loss: float
    iteration: int


def _run_batch(
    vocab: Vocabulary,
    config: StackedTrainingConfig,
    batch: Tuple[Tuple[int, ...], Tuple[int, ...]],
    state: StackedTrainerState,
) -> StackedTrainerState:
    input_ids, target_ids = batch
    inputs = [one_hot(vocab, index) for index in input_ids]
    hs_by_layer, _, ps = stacked_forward_sequence(state.params, inputs, state.hidden)
    loss = cross_entropy_loss(ps, target_ids)
    grads = stacked_backpropagate_through_time(
        state.params, inputs, target_ids, hs_by_layer, ps
    )
    grads = clip_stacked_gradients(grads)
    new_params, new_memory = stacked_adagrad_update(
        state.params, grads, state.memory, config.learning_rate
    )
    return StackedTrainerState(
        params=new_params,
        memory=new_memory,
        hidden=tuple(layer_states[-1] for layer_states in hs_by_layer),
        smooth_loss=state.smooth_loss * 0.999 + loss * 0.001,
        iteration=state.iteration + 1,
    )


def _run_epoch(
    vocab: Vocabulary,
    config: StackedTrainingConfig,
    indices: Tuple[int, ...],
    state: StackedTrainerState,
) -> Tuple[StackedTrainerState, Tuple[StackedTrainingSnapshot, ...]]:
    state = replace(state, hidden=tuple(np.zeros_like(hidden) for hidden in state.hidden))
    snapshots = ()
    for batch in make_character_batches(indices, config.seq_length):
        state = _run_batch(vocab, config, batch, state)
        snapshots += (
            StackedTrainingSnapshot(state.iteration, state.smooth_loss, state.params),
        )
    return state, snapshots


def train_stacked_rnn(
    corpus: str,
    vocab: Vocabulary,
    config: StackedTrainingConfig,
    epochs: int,
) -> StackedTrainingResult:
    """Train a stacked vanilla RNN over shifted character windows."""
    indices = encode(vocab, corpus)
    params = initialize_stacked_rnn_parameters(
        vocab.size, config.hidden_sizes, config.seed
    )
    state = StackedTrainerState(
        params=params,
        memory=zero_stacked_memory(params),
        hidden=tuple(np.zeros((size, 1)) for size in config.hidden_sizes),
        smooth_loss=-np.log(1.0 / vocab.size) * config.seq_length,
        iteration=0,
    )
    snapshots = ()
    for _ in range(epochs):
        state, epoch_snapshots = _run_epoch(vocab, config, indices, state)
        snapshots += epoch_snapshots
    return StackedTrainingResult(params=state.params, snapshots=snapshots)