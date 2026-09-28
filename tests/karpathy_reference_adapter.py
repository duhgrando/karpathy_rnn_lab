"""Functional Karpathy RNN reference with an application-shaped adapter.

Adapted to Python 3 from https://gist.github.com/karpathy/d4dee566867f8291f086.
The reference functions copy their initial parameters and return the same
TrainingResult shape as the application trainer.
"""
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from application.train_service import TrainingConfig, TrainingResult, TrainingSnapshot
from domain.rnn_model import (
    HiddenState,
    LogitOutput,
    ProbabilityOutput,
    RNNParams,
    RNNOutput,
)
from domain.rnn_types import RNNArray, RNNInput
from domain.vocabulary import Vocabulary, encode

PARAMETER_NAMES: tuple[str, ...] = ("Wxh", "Whh", "Why", "bh", "by")
FloatArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]
ParameterArrays: TypeAlias = dict[str, FloatArray]
TimeArrays: TypeAlias = tuple[FloatArray, ...]


def _copy_parameters(params: RNNParams) -> ParameterArrays:
    """Copy model arrays before the reference update loop begins."""
    return {name: getattr(params, name).copy() for name in PARAMETER_NAMES}


def _snapshot_parameters(params: ParameterArrays) -> RNNParams:
    """Build independent domain parameters from reference arrays."""
    return RNNParams(**{name: params[name].copy() for name in PARAMETER_NAMES})


def forward_reference_step(
    params: RNNParams, x_t: RNNInput, h_prev: RNNArray
) -> RNNOutput:
    """Compute Karpathy's step equations using the shared domain output type."""
    hidden_state = HiddenState(
        np.tanh(params.Whh @ h_prev + params.Wxh @ x_t + params.bh)
    )
    logits = LogitOutput(params.Why @ hidden_state + params.by)
    probabilities = ProbabilityOutput(np.exp(logits) / np.sum(np.exp(logits)))
    return RNNOutput(hidden_state, logits, probabilities)


def _forward(
    params: ParameterArrays,
    inputs: list[int],
    targets: list[int],
    hidden: FloatArray,
    vocab_size: int,
) -> tuple[TimeArrays, TimeArrays, TimeArrays, float]:
    """Compute one-hot states, probabilities, and summed cross-entropy."""
    xs: TimeArrays = ()
    hs: TimeArrays = (hidden.copy(),)
    ps: TimeArrays = ()
    loss: float = 0.0
    for input_index, target in zip(inputs, targets):
        x: FloatArray = np.eye(vocab_size)[:, input_index:input_index + 1]
        hidden_state: FloatArray = np.tanh(
            params["Wxh"] @ x + params["Whh"] @ hs[-1] + params["bh"]
        )
        logits: FloatArray = params["Why"] @ hidden_state + params["by"]
        probabilities: FloatArray = np.exp(logits) / np.sum(np.exp(logits))
        xs = xs + (x,)
        hs = hs + (hidden_state,)
        ps = ps + (probabilities,)
        loss = loss - np.log(probabilities[target, 0])
    return xs, hs, ps, loss


def _backward(
    params: ParameterArrays,
    inputs: list[int],
    targets: list[int],
    xs: TimeArrays,
    hs: TimeArrays,
    ps: TimeArrays,
    hidden: FloatArray,
) -> ParameterArrays:
    """Accumulate gradients for the sequence loss using reverse-time BPTT."""
    gradients: ParameterArrays = {
        name: np.zeros_like(params[name]) for name in PARAMETER_NAMES
    }
    dh_next: FloatArray = np.zeros_like(hidden)
    for t in reversed(range(len(inputs))):
        dy: FloatArray = ps[t] - np.eye(ps[t].shape[0])[:, targets[t]:targets[t] + 1]
        hidden_state = hs[t + 1]
        dh: FloatArray = params["Why"].T @ dy + dh_next
        dh_raw: FloatArray = (1.0 - hidden_state * hidden_state) * dh
        contributions: ParameterArrays = {
            "Wxh": dh_raw @ xs[t].T,
            "Whh": dh_raw @ hs[t].T,
            "Why": dy @ hidden_state.T,
            "bh": dh_raw,
            "by": dy,
        }
        gradients = {
            name: gradients[name] + contributions[name]
            for name in PARAMETER_NAMES
        }
        dh_next = params["Whh"].T @ dh_raw
    return gradients


def _apply_adagrad(
    params: ParameterArrays,
    gradients: ParameterArrays,
    memory: ParameterArrays,
    learning_rate: float,
) -> tuple[ParameterArrays, ParameterArrays]:
    """Return clipped Adagrad updates without mutating parameters or memory."""
    clipped_gradients = {
        name: np.clip(gradients[name], -5.0, 5.0) for name in PARAMETER_NAMES
    }
    new_memory = {
        name: memory[name] + clipped_gradients[name] * clipped_gradients[name]
        for name in PARAMETER_NAMES
    }
    new_params = {
        name: params[name]
        - learning_rate * clipped_gradients[name] / np.sqrt(new_memory[name] + 1e-8)
        for name in PARAMETER_NAMES
    }
    return new_params, new_memory


def _training_step(
    params: ParameterArrays,
    memory: ParameterArrays,
    hidden: FloatArray,
    inputs: list[int],
    targets: list[int],
    vocab_size: int,
    learning_rate: float,
) -> tuple[ParameterArrays, ParameterArrays, FloatArray, float]:
    """Run a reference chunk and return new optimizer and recurrent state."""
    xs, hs, ps, loss = _forward(params, inputs, targets, hidden, vocab_size)
    gradients = _backward(params, inputs, targets, xs, hs, ps, hidden)
    new_params, new_memory = _apply_adagrad(
        params, gradients, memory, learning_rate
    )
    return new_params, new_memory, hs[-1], loss


def train_reference(
    corpus: str,
    vocab: Vocabulary,
    config: TrainingConfig,
    initial_params: RNNParams,
    epochs: int,
) -> TrainingResult:
    """Train the reference over sequential chunks without mutating inputs."""
    data: list[int] = list(encode(vocab, corpus))
    params: ParameterArrays = _copy_parameters(initial_params)
    memory: ParameterArrays = {
        name: np.zeros_like(value) for name, value in params.items()
    }
    hidden: FloatArray = np.zeros((config.hidden_size, 1))
    smooth_loss: float = -np.log(1.0 / vocab.size) * config.seq_length
    snapshots: tuple[TrainingSnapshot, ...] = ()
    iteration: int = 0

    for _ in range(epochs):
        hidden = np.zeros_like(hidden)
        for start in range(0, len(data) - config.seq_length - 1, config.seq_length):
            inputs: list[int] = data[start:start + config.seq_length]
            targets: list[int] = data[start + 1:start + config.seq_length + 1]
            params, memory, hidden, loss = _training_step(
                params,
                memory,
                hidden,
                inputs,
                targets,
                vocab.size,
                config.learning_rate,
            )
            iteration += 1
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            snapshots = snapshots + (
                TrainingSnapshot(
                    iteration=iteration,
                    smooth_loss=smooth_loss,
                    params=_snapshot_parameters(params),
                ),
            )

    return TrainingResult(params=_snapshot_parameters(params), snapshots=snapshots)


def next_character_probabilities(
    params: RNNParams,
    vocab: Vocabulary,
    input_ids: tuple[int, ...],
) -> tuple[ProbabilityOutput, ...]:
    """Return next-character distributions for fixed inputs from a zero state."""
    hidden: FloatArray = np.zeros((params.hidden_size, 1))
    probabilities: tuple[ProbabilityOutput, ...] = ()
    for input_index in input_ids:
        x: FloatArray = np.eye(vocab.size)[:, input_index:input_index + 1]
        step_output = forward_reference_step(params, x, hidden)
        hidden = step_output.hidden_state
        probabilities = probabilities + (step_output.probabilities,)
    return probabilities


@dataclass(frozen=True)
class KarpathyReferenceAdapter:
    """Bind shared initial parameters to the application's training interface."""

    initial_params: RNNParams

    def train(
        self,
        corpus: str,
        vocab: Vocabulary,
        config: TrainingConfig,
        epochs: int,
    ) -> TrainingResult:
        """Match the application trainer's arguments and result type."""
        return train_reference(corpus, vocab, config, self.initial_params, epochs)

    @staticmethod
    def forward_step(
        params: RNNParams, x_t: RNNInput, h_prev: RNNArray
    ) -> RNNOutput:
        """Expose the reference step through the shared domain output type."""
        return forward_reference_step(params, x_t, h_prev)

    @staticmethod
    def next_character_probabilities(
        params: RNNParams,
        vocab: Vocabulary,
        input_ids: tuple[int, ...],
    ) -> tuple[ProbabilityOutput, ...]:
        """Expose reference inference through the adapter interface."""
        return next_character_probabilities(params, vocab, input_ids)