"""Application service for training the character-level LSTM."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np

from application.train_service import TrainingConfig
from application.training_utils import make_character_batches
from domain.lstm_model import (
    LSTMParams,
    initialize_lstm_parameters,
    lstm_forward_sequence,
)
from domain.lstm_training import (
    LSTMMemory,
    lstm_backpropagate_through_time,
)
from domain.optimization import adagrad_update, clip_gradients, zero_memory
from domain.rnn_model import cross_entropy_loss
from domain.vocabulary import Vocabulary, encode, one_hot

__all__ = [
    "LSTMTrainingSnapshot",
    "LSTMTrainingResult",
    "LSTMTrainerState",
    "train_lstm",
]


@dataclass(frozen=True)
class LSTMTrainingSnapshot:
    iteration: int
    smooth_loss: float
    params: LSTMParams


@dataclass(frozen=True)
class LSTMTrainingResult:
    params: LSTMParams
    snapshots: Tuple[LSTMTrainingSnapshot, ...]


@dataclass(frozen=True)
class LSTMTrainerState:
    params: LSTMParams
    memory: LSTMMemory
    hidden: np.ndarray
    cell: np.ndarray
    smooth_loss: float
    iteration: int


def _run_batch(
    vocab: Vocabulary,
    config: TrainingConfig,
    batch: Tuple[Tuple[int, ...], Tuple[int, ...]],
    state: LSTMTrainerState,
) -> LSTMTrainerState:
    input_ids, target_ids = batch
    inputs = [one_hot(vocab, index) for index in input_ids]
    hs, cs, _, ps = lstm_forward_sequence(
        state.params, inputs, state.hidden, state.cell
    )
    loss = cross_entropy_loss(ps, target_ids)
    grads, _, _ = lstm_backpropagate_through_time(
        state.params, inputs, target_ids, hs, cs, ps
    )
    grads = clip_gradients(grads)
    new_params, new_memory = adagrad_update(
        state.params, grads, state.memory, config.learning_rate
    )
    return LSTMTrainerState(
        params=new_params,
        memory=new_memory,
        hidden=hs[-1],
        cell=cs[-1],
        smooth_loss=state.smooth_loss * 0.999 + loss * 0.001,
        iteration=state.iteration + 1,
    )


def _run_epoch(
    vocab: Vocabulary,
    config: TrainingConfig,
    indices: Tuple[int, ...],
    state: LSTMTrainerState,
) -> Tuple[LSTMTrainerState, Tuple[LSTMTrainingSnapshot, ...]]:
    state = replace(
        state, hidden=np.zeros_like(state.hidden), cell=np.zeros_like(state.cell)
    )
    snapshots = ()
    for batch in make_character_batches(indices, config.seq_length):
        state = _run_batch(vocab, config, batch, state)
        snapshots += (
            LSTMTrainingSnapshot(state.iteration, state.smooth_loss, state.params),
        )
    return state, snapshots


def train_lstm(
    corpus: str,
    vocab: Vocabulary,
    config: TrainingConfig,
    epochs: int,
) -> LSTMTrainingResult:
    """Train an LSTM over shifted character windows and return batch snapshots."""
    indices = encode(vocab, corpus)
    params = initialize_lstm_parameters(vocab.size, config.hidden_size, config.seed)
    state = LSTMTrainerState(
        params=params,
        memory=zero_memory(params, LSTMMemory),
        hidden=np.zeros((config.hidden_size, 1)),
        cell=np.zeros((config.hidden_size, 1)),
        smooth_loss=-np.log(1.0 / vocab.size) * config.seq_length,
        iteration=0,
    )
    snapshots = ()
    for _ in range(epochs):
        state, epoch_snapshots = _run_epoch(vocab, config, indices, state)
        snapshots += epoch_snapshots
    return LSTMTrainingResult(params=state.params, snapshots=snapshots)