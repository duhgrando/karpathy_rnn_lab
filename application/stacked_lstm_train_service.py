"""Application service for minibatched training of stacked LSTM layers."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator, Sequence, Tuple

import numpy as np

from domain.stacked_lstm import (
    StackedLSTMParams,
    initialize_stacked_lstm_parameters,
    stacked_lstm_forward_sequence,
)
from domain.stacked_lstm_training import (
    StackedLSTMGradients,
    StackedLSTMMemory,
    clip_stacked_lstm_gradients,
    stacked_lstm_adagrad_update,
    stacked_lstm_backpropagate_through_time,
    stacked_lstm_cross_entropy_loss,
    zero_stacked_lstm_memory,
)
from domain.vocabulary import Vocabulary, encode, one_hot

__all__ = [
    "StackedLSTMTrainingConfig",
    "StackedLSTMTrainingSnapshot",
    "StackedLSTMTrainingResult",
    "StackedLSTMTrainerState",
    "train_stacked_lstm",
]


@dataclass(frozen=True)
class StackedLSTMTrainingConfig:
    hidden_sizes: Tuple[int, ...] = (100, 100)
    seq_length: int = 25
    learning_rate: float = 0.1
    seed: int = 0
    batch_size: int = 1
    dropout_probability: float = 0.0


@dataclass(frozen=True)
class StackedLSTMTrainingSnapshot:
    iteration: int
    smooth_loss: float
    params: StackedLSTMParams


@dataclass(frozen=True)
class StackedLSTMTrainingResult:
    params: StackedLSTMParams
    snapshots: Tuple[StackedLSTMTrainingSnapshot, ...]


@dataclass(frozen=True)
class StackedLSTMTrainerState:
    params: StackedLSTMParams
    memory: StackedLSTMMemory
    hidden: Tuple[np.ndarray, ...]
    cell: Tuple[np.ndarray, ...]
    smooth_loss: float
    iteration: int


def _make_minibatches(
    indices: Tuple[int, ...], seq_length: int, batch_size: int
) -> Iterator[Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]]:
    """Split a corpus into fixed contiguous streams and batch aligned windows."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    stream_length = len(indices) // batch_size
    if stream_length <= seq_length:
        return
    streams = _split_streams(indices, batch_size, stream_length)
    for offset in range(0, stream_length - seq_length, seq_length):
        yield _aligned_window(streams, offset, seq_length)


def _split_streams(indices, batch_size, stream_length):
    return tuple(
        indices[index * stream_length:(index + 1) * stream_length]
        for index in range(batch_size)
    )


def _aligned_window(streams, offset, seq_length):
    pairs = tuple(_window_timestep(streams, offset, timestep) for timestep in range(seq_length))
    inputs, targets = zip(*pairs)
    return tuple(inputs), tuple(targets)


def _window_timestep(streams, offset, timestep):
    inputs = tuple(stream[offset + timestep] for stream in streams)
    targets = tuple(stream[offset + timestep + 1] for stream in streams)
    return inputs, targets


def _one_hot_batch(vocab: Vocabulary, time_step_ids: Sequence[int]) -> np.ndarray:
    return np.column_stack([one_hot(vocab, index) for index in time_step_ids])


def _run_batch(
    vocab: Vocabulary,
    config: StackedLSTMTrainingConfig,
    batch: Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]],
    state: StackedLSTMTrainerState,
) -> StackedLSTMTrainerState:
    input_ids, target_ids = batch
    inputs = tuple(_one_hot_batch(vocab, time_step) for time_step in input_ids)
    targets = np.asarray(target_ids, dtype=np.intp)
    hs, cs, _, probabilities, masks = stacked_lstm_forward_sequence(
        state.params,
        inputs,
        state.hidden,
        state.cell,
        dropout_probability=config.dropout_probability,
        training=config.dropout_probability > 0.0,
        seed=config.seed + state.iteration,
    )
    loss = stacked_lstm_cross_entropy_loss(probabilities, targets)
    gradients = stacked_lstm_backpropagate_through_time(
        state.params, inputs, targets, hs, cs, probabilities, masks
    )
    gradients = clip_stacked_lstm_gradients(gradients)
    new_params, new_memory = stacked_lstm_adagrad_update(
        state.params, gradients, state.memory, config.learning_rate
    )
    return StackedLSTMTrainerState(
        params=new_params,
        memory=new_memory,
        hidden=tuple(layer_states[-1] for layer_states in hs),
        cell=tuple(layer_states[-1] for layer_states in cs),
        smooth_loss=state.smooth_loss * 0.999 + loss * 0.001,
        iteration=state.iteration + 1,
    )


def _run_epoch(
    vocab: Vocabulary,
    config: StackedLSTMTrainingConfig,
    indices: Tuple[int, ...],
    state: StackedLSTMTrainerState,
) -> Tuple[StackedLSTMTrainerState, Tuple[StackedLSTMTrainingSnapshot, ...]]:
    state = replace(
        state,
        hidden=tuple(np.zeros_like(hidden) for hidden in state.hidden),
        cell=tuple(np.zeros_like(cell) for cell in state.cell),
    )
    snapshots = ()
    for batch in _make_minibatches(indices, config.seq_length, config.batch_size):
        state = _run_batch(vocab, config, batch, state)
        snapshots += (
            StackedLSTMTrainingSnapshot(state.iteration, state.smooth_loss, state.params),
        )
    return state, snapshots


def _validate_batch_config(config):
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.seq_length <= 0:
        raise ValueError("seq_length must be positive")


def _validate_run_config(config, epochs):
    if epochs < 0:
        raise ValueError("epochs must be nonnegative")
    if not 0.0 <= config.dropout_probability < 1.0:
        raise ValueError("dropout_probability must be in [0, 1)")


def train_stacked_lstm(
    corpus: str,
    vocab: Vocabulary,
    config: StackedLSTMTrainingConfig,
    epochs: int,
) -> StackedLSTMTrainingResult:
    """Train a stacked LSTM over aligned contiguous character streams."""
    _validate_batch_config(config)
    _validate_run_config(config, epochs)
    indices = encode(vocab, corpus)
    params = initialize_stacked_lstm_parameters(
        vocab.size, config.hidden_sizes, config.seed
    )
    hidden = tuple(np.zeros((size, config.batch_size)) for size in config.hidden_sizes)
    state = StackedLSTMTrainerState(
        params=params,
        memory=zero_stacked_lstm_memory(params),
        hidden=hidden,
        cell=tuple(np.zeros_like(layer_hidden) for layer_hidden in hidden),
        smooth_loss=-np.log(1.0 / vocab.size),
        iteration=0,
    )
    snapshots = ()
    for _ in range(epochs):
        state, epoch_snapshots = _run_epoch(vocab, config, indices, state)
        snapshots += epoch_snapshots
    return StackedLSTMTrainingResult(params=state.params, snapshots=snapshots)