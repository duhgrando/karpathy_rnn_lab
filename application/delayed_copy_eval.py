"""Application: delayed-copy evaluation for the RNN and LSTM baselines."""
from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter
from typing import Sequence, Tuple

import numpy as np

from application.lstm_train_service import train_lstm
from application.train_service import TrainingConfig, train
from domain.lstm_model import lstm_forward_sequence
from domain.rnn_model import forward_sequence
from domain.vocabulary import build_vocabulary, decode, encode, one_hot
from infrastructure.delayed_copy import DelayedCopyExample


@dataclass(frozen=True)
class DelayedCopyEvaluation:
    model_name: str
    train_loss: float
    eval_loss: float
    eval_accuracy: float
    answer_count: int


def answer_only_loss_and_accuracy(
    probabilities: Sequence[np.ndarray] | np.ndarray,
    answer_mask: Sequence[bool] | np.ndarray,
    answer_targets: Sequence[int],
) -> Tuple[float, float]:
    """Compute cross-entropy and accuracy only on the delayed-copy answer span."""
    mask = np.asarray(answer_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("answer_mask must be a 1D boolean array")

    probability_array = np.atleast_2d(np.asarray(probabilities, dtype=float))
    prob_rows = probability_array.reshape(probability_array.shape[0], -1)

    if prob_rows.shape[0] != len(mask):
        raise ValueError("probabilities and answer_mask lengths must match")

    targets = np.asarray(answer_targets, dtype=int)
    if len(targets) != int(np.sum(mask)):
        raise ValueError("answer_targets length must equal the number of masked positions")

    answer_rows = np.flatnonzero(mask)
    selected_probabilities = prob_rows[answer_rows, targets]
    answer_count = max(len(targets), 1)
    loss = np.sum(-np.log(selected_probabilities + 1e-12)) / answer_count
    predictions = np.argmax(prob_rows[answer_rows], axis=1)
    accuracy = np.sum(predictions == targets) / answer_count
    return float(loss), float(accuracy)


def _shared_vocabulary(examples: Sequence[DelayedCopyExample]) -> object:
    vocab_text = "".join(example.sequence for example in examples)
    return build_vocabulary(vocab_text)


def _evaluate_example(
    model_name: str,
    params,
    example: DelayedCopyExample,
    hidden_size: int,
    vocab,
):
    indices = encode(vocab, example.sequence)
    inputs = [one_hot(vocab, index) for index in indices]
    if model_name == "rnn":
        _, _, probabilities = forward_sequence(params, inputs, np.zeros((hidden_size, 1)))
    elif model_name == "lstm":
        _, _, _, probabilities = lstm_forward_sequence(
            params,
            inputs,
            np.zeros((hidden_size, 1)),
            np.zeros((hidden_size, 1)),
        )
    else:
        raise ValueError("model_name must be 'rnn' or 'lstm'")
    prediction_mask = np.roll(example.answer_mask, -1)
    answer_targets = encode(vocab, example.payload)
    loss, accuracy = answer_only_loss_and_accuracy(
        probabilities, prediction_mask, answer_targets
    )
    probability_rows = np.asarray(probabilities).reshape(len(prediction_mask), -1)
    predicted_tokens = decode(
        vocab,
        tuple(map(int, np.argmax(probability_rows, axis=1))),
    )
    return loss, accuracy, predicted_tokens


def _model_result(train_result, model_name: str, examples: Sequence[DelayedCopyExample], hidden_size: int, vocab):
    losses = []
    accuracies = []
    predictions = []
    for example in examples:
        loss, accuracy, prediction = _evaluate_example(
            model_name, train_result.params, example, hidden_size, vocab
        )
        losses.append(loss)
        accuracies.append(accuracy)
        predictions.append(prediction)
    return float(np.mean(losses)), float(np.mean(accuracies)), tuple(predictions)


def evaluate_delayed_copy(
    model_name: str,
    train_examples: Sequence[DelayedCopyExample],
    eval_examples: Sequence[DelayedCopyExample],
    hidden_size: int = 8,
    epochs: int = 8,
    seed: int = 0,
    learning_rate: float = 0.1,
    seq_length: int = 8,
) -> dict:
    """Train a character-level delayed-copy model and score only answer positions."""
    trainers = {"rnn": train, "lstm": train_lstm}
    if model_name not in trainers:
        raise ValueError("model_name must be 'rnn' or 'lstm'")
    if not train_examples:
        raise ValueError("train_examples must not be empty")

    vocab = _shared_vocabulary(tuple(train_examples) + tuple(eval_examples))
    corpus = "".join(map(attrgetter("sequence"), train_examples))
    config = TrainingConfig(
        hidden_size=hidden_size,
        seq_length=seq_length,
        learning_rate=learning_rate,
        seed=seed,
    )
    train_result = trainers[model_name](corpus, vocab, config, epochs)

    train_loss, _, _ = _model_result(train_result, model_name, train_examples, hidden_size, vocab)
    eval_loss, eval_accuracy, eval_predictions = _model_result(
        train_result, model_name, eval_examples, hidden_size, vocab
    )
    return {
        "model_name": model_name,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "eval_accuracy": eval_accuracy,
        "answer_count": sum(map(len, map(attrgetter("answer_tokens"), eval_examples))),
        "eval_predictions": eval_predictions,
    }
