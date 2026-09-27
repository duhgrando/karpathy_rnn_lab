import sys

from application.delayed_copy_eval import evaluate_delayed_copy
from infrastructure.delayed_copy import generate_delayed_copy_dataset
import scripts.run_delayed_copy_demo as run_delayed_copy_demo


def _metrics_are_valid(result):
    return all((
        result["train_loss"] >= 0.0,
        result["eval_loss"] >= 0.0,
        0.0 <= result["eval_accuracy"] <= 1.0,
        result["answer_count"] > 0,
    ))


def test_delayed_copy_training_produces_finite_metrics_for_rnn_and_lstm():
    train_examples, eval_examples = generate_delayed_copy_dataset(
        ["ab", "cd", "ef", "gh", "ij"],
        delay=3,
        seed=3,
    )

    rnn_result = evaluate_delayed_copy(
        model_name="rnn",
        train_examples=train_examples,
        eval_examples=eval_examples,
        hidden_size=8,
        epochs=8,
        seed=7,
    )
    lstm_result = evaluate_delayed_copy(
        model_name="lstm",
        train_examples=train_examples,
        eval_examples=eval_examples,
        hidden_size=8,
        epochs=8,
        seed=7,
    )

    assert all(map(_metrics_are_valid, (rnn_result, lstm_result)))


def test_mixed_payload_baseline_learns_the_delayed_copy_for_both_models():
    train_examples, eval_examples = generate_delayed_copy_dataset(
        (["ab"] * 3 + ["ef"] * 2) * 3,
        delay=3,
        seed=7,
    )

    for model_name in ("rnn", "lstm"):
        result = evaluate_delayed_copy(
            model_name=model_name,
            train_examples=train_examples,
            eval_examples=eval_examples,
            hidden_size=8,
            epochs=8,
            seed=7,
        )
        assert result["eval_accuracy"] > 0.0


def test_demo_trace_wraps_end_to_start(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_delayed_copy_demo.py", "--payloads", "ab"])

    def circular_predictions(model_name, train_examples, eval_examples, **kwargs):
        sequence = eval_examples[0].sequence
        return {
            "train_loss": 0.0,
            "eval_loss": 0.0,
            "eval_accuracy": 1.0,
            "eval_predictions": (sequence[1:] + sequence[:1],),
        }

    monkeypatch.setattr(run_delayed_copy_demo, "evaluate_delayed_copy", circular_predictions)
    run_delayed_copy_demo.main()

    assert "'<end>' -> '<start>', expected '<start>'" in capsys.readouterr().out
