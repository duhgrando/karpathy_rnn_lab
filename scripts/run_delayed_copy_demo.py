"""Run a tiny delayed-copy demonstration for the vanilla RNN and LSTM."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.delayed_copy_eval import evaluate_delayed_copy
from infrastructure.delayed_copy import (
    END_TOKEN,
    START_TOKEN,
    generate_delayed_copy_dataset,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate delayed-copy baselines.")
    parser.add_argument(
        "--payloads",
        nargs="*",
        default=["ab", "ab", "ab", "ef", "ef"] * 3,
    )
    parser.add_argument("--delay", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def run_demo(payloads, delay, hidden_size, epochs, seed):
    train_examples, eval_examples = generate_delayed_copy_dataset(
        payloads,
        delay,
        seed=seed,
    )
    model_results = tuple(
        (
            model_name,
            evaluate_delayed_copy(
                model_name=model_name,
                train_examples=train_examples,
                eval_examples=eval_examples,
                hidden_size=hidden_size,
                epochs=epochs,
                seed=seed,
            ),
        )
        for model_name in ("rnn", "lstm")
    )
    return eval_examples, model_results


def _display_token(token, token_names):
    return token_names.get(token, token)


def _display_sequence(sequence, token_names):
    return "".join(_display_token(token, token_names) for token in sequence)


def _format_prediction_trace(example, prediction, token_names):
    expected_tokens = example.sequence[1:] + example.sequence[:1]
    return (
        f"  sequence: {_display_sequence(example.sequence, token_names)}",
        *tuple(
            f"    {_display_token(input_token, token_names)!r} -> "
            f"{_display_token(predicted_token, token_names)!r}, "
            f"expected {_display_token(expected_token, token_names)!r}"
            for input_token, predicted_token, expected_token in zip(
                example.sequence, prediction, expected_tokens
            )
        )
    )


def _format_model_result(model_name, result, eval_examples, token_names):
    return (
        f"{model_name}: train_loss={result['train_loss']:.4f} "
        f"eval_loss={result['eval_loss']:.4f} eval_accuracy={result['eval_accuracy']:.4f}",
        "  next-token trace (input -> predicted, expected):",
        *tuple(
            line
            for example, prediction in zip(eval_examples, result["eval_predictions"])
            for line in _format_prediction_trace(example, prediction, token_names)
        ),
    )


def format_demo(delay, hidden_size, epochs, seed, eval_examples, model_results):
    token_names = {START_TOKEN: "<start>", END_TOKEN: "<end>"}
    return (
        f"delay={delay} hidden_size={hidden_size} epochs={epochs} seed={seed}",
        *tuple(
            line
            for model_name, result in model_results
            for line in _format_model_result(
                model_name, result, eval_examples, token_names
            )
        ),
    )


def main() -> None:
    args = _build_parser().parse_args()
    eval_examples, model_results = run_demo(
        args.payloads,
        args.delay,
        args.hidden_size,
        args.epochs,
        args.seed,
    )
    print("\n".join(format_demo(
        args.delay,
        args.hidden_size,
        args.epochs,
        args.seed,
        eval_examples,
        model_results,
    )))


if __name__ == "__main__":
    main()
