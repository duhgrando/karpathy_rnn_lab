"""Inspect a tiny RNN trained on the checked-in Shakespeare corpus."""
from __future__ import annotations

import argparse
import sys
from functools import partial
from operator import itemgetter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from application.stacked_lstm_train_service import (
    StackedLSTMTrainingConfig,
    train_stacked_lstm,
)
from domain.sampling import sample_stacked_lstm
from domain.stacked_lstm import stacked_lstm_forward_sequence
from domain.stacked_lstm_training import stacked_lstm_cross_entropy_loss
from domain.vocabulary import build_vocabulary, encode
from infrastructure.corpus import load_corpus


DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[1]
    / "infrastructure"
    / "input_tinyshakespeare.txt"
)
DEFAULT_PROMPT = "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\n"
COMPARISON_BLOCK_SIZE = 80


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a character LSTM on Tiny Shakespeare and inspect its output."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument(
        "--characters",
        type=int,
        default=10000,
        help="optional corpus prefix limit; default trains on the full file",
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seq-length", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--validation-window", type=int, default=100)
    parser.add_argument("--sample-length", type=int, default=100)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="sampling randomness; lower values are more conservative",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _decode(vocab, indices) -> str:
    return "".join(map(vocab.chars.__getitem__, indices))


def _format_comparison_block(vocab, expected_indices, predicted_indices, start):
    end = start + COMPARISON_BLOCK_SIZE
    return "\n".join((
        f"\n[block: {start + 1}-{min(end, len(expected_indices))}]",
        "[expected]",
        repr(_decode(vocab, expected_indices[start:end])),
        "[predicted]",
        repr(_decode(vocab, predicted_indices[start:end])),
    ))


def _print_comparison_blocks(vocab, input_indices, expected_indices, predicted_indices):
    print("\nValidation context (characters supplied to the model):")
    print(_decode(vocab, input_indices))
    print("\nExpected vs. predicted next characters (teacher-forced argmax):")
    format_block = partial(
        _format_comparison_block, vocab, expected_indices, predicted_indices
    )
    starts = range(0, len(expected_indices), COMPARISON_BLOCK_SIZE)
    print("\n".join(map(format_block, starts)))


def _run_validation_evaluation(
    trained, vocab, validation_indices, requested_window, hidden_size, identity
):
    validation_window = min(requested_window, len(validation_indices) - 1)
    validation_inputs = identity[
        np.asarray(validation_indices[:validation_window])
    ][:, :, np.newaxis]
    validation_targets = np.asarray(
        validation_indices[1:validation_window + 1], dtype=np.intp
    )[:, np.newaxis]
    initial_states = (np.zeros((hidden_size, 1)),) * 2
    _, _, _, validation_probabilities, _ = stacked_lstm_forward_sequence(
        trained.params,
        validation_inputs,
        initial_states,
        initial_states,
    )
    validation_loss = stacked_lstm_cross_entropy_loss(
        validation_probabilities, validation_targets
    )
    validation_predictions = np.argmax(
        np.column_stack(validation_probabilities), axis=0
    )
    validation_correct = int(
        np.sum(validation_predictions == validation_targets[:, 0])
    )

    print("\n=== HELD-OUT VALIDATION ===")
    print("Next-character predictions; the real preceding characters are supplied.")
    print(
        f"loss={validation_loss:.4f}; accuracy={validation_correct}/"
        f"{validation_window} ({validation_correct / validation_window:.1%})"
    )
    _print_comparison_blocks(
        vocab,
        validation_indices[:validation_window],
        validation_targets[:, 0],
        validation_predictions,
    )


def _run_prompt_generation(
    trained, vocab, prompt, sample_length, temperature, seed, hidden_size, identity
):
    prompt_indices = encode(vocab, prompt)
    prompt_inputs = identity[np.asarray(prompt_indices[:-1])][:, :, np.newaxis]
    initial_states = (np.zeros((hidden_size, 1)),) * 2
    primed_hidden, primed_cell, _, _, _ = stacked_lstm_forward_sequence(
        trained.params, prompt_inputs, initial_states, initial_states
    )
    generated = sample_stacked_lstm(
        trained.params,
        vocab,
        tuple(map(itemgetter(-1), primed_hidden)),
        tuple(map(itemgetter(-1), primed_cell)),
        seed_index=prompt_indices[-1],
        length=sample_length,
        temperature=temperature,
        seed=seed,
    )

    print("\n=== PROMPT-BASED GENERATION ===")
    print("Free-running sample; generated characters become the next context.")
    print(f"Prompt: {prompt!r}")
    print(f"Generated continuation (temperature={temperature}):")
    print(_decode(vocab, generated))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    corpus = load_corpus(str(args.corpus))[:args.characters]
    split_index = int(len(corpus) * 0.95)
    train_corpus = corpus[:split_index]
    validation_corpus = corpus[split_index:]
    vocab = build_vocabulary(corpus)
    config = StackedLSTMTrainingConfig(
        hidden_sizes=(args.hidden_size, args.hidden_size),
        seq_length=args.seq_length,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    trained = train_stacked_lstm(train_corpus, vocab, config, epochs=args.epochs)

    validation_indices = encode(vocab, validation_corpus)
    identity = np.eye(vocab.size)

    print(f"corpus: {args.corpus}")
    print(f"characters: {len(corpus)}; vocabulary: {vocab.size}")
    print(f"training/validation split: {len(train_corpus)}/{len(validation_corpus)} characters")
    print(
        f"model: 2-layer LSTM hidden_size={args.hidden_size} "
        f"batch_size={args.batch_size} seq_length={args.seq_length} "
        f"epochs={args.epochs} seed={args.seed}"
    )
    print(
        f"training: {len(trained.snapshots)} updates; "
        f"final loss EMA={trained.snapshots[-1].smooth_loss:.4f}"
    )
    _run_validation_evaluation(
        trained,
        vocab,
        validation_indices,
        args.validation_window,
        args.hidden_size,
        identity,
    )
    _run_prompt_generation(
        trained,
        vocab,
        args.prompt,
        args.sample_length,
        args.temperature,
        args.seed,
        args.hidden_size,
        identity,
    )


if __name__ == "__main__":
    main()
