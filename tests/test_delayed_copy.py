import numpy as np

import application.delayed_copy_eval as delayed_copy_eval
from application.delayed_copy_eval import (
    _evaluate_example,
    answer_only_loss_and_accuracy,
)
from domain.vocabulary import build_vocabulary, encode
from infrastructure.delayed_copy import (
    END_TOKEN,
    START_TOKEN,
    DelayedCopyExample,
    build_delayed_copy_example,
    generate_delayed_copy_dataset,
)


def test_delayed_copy_example_encodes_prompt_and_delay():
    example = build_delayed_copy_example("ab", 3, distractor=".")

    assert isinstance(example, DelayedCopyExample)
    assert example.sequence == f"{START_TOKEN}!ab#...=ab{END_TOKEN}"


def test_delayed_copy_example_marks_only_the_answer_span():
    example = build_delayed_copy_example("ab", 3, distractor=".")

    assert example.answer_mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
    ]
    assert example.answer_tokens == ("a", "b")


def test_delayed_copy_example_exposes_prompt_and_delay():
    example = build_delayed_copy_example("ab", 3, distractor=".")

    assert example.prompt == f"{START_TOKEN}!ab#"
    assert example.delay == 3


def test_delayed_copy_dataset_is_reproducible_for_the_same_seed():
    train_a, eval_a = generate_delayed_copy_dataset(["ab", "cd", "ef"], 2, seed=10)
    train_b, eval_b = generate_delayed_copy_dataset(["ab", "cd", "ef"], 2, seed=10)

    assert train_a == train_b
    assert eval_a == eval_b


def test_delayed_copy_dataset_reuses_the_only_payload_for_evaluation():
    train_examples, eval_examples = generate_delayed_copy_dataset(["ab"], 2, seed=10)

    assert len(train_examples) == 1
    assert len(eval_examples) == 1
    assert train_examples[0] == eval_examples[0]


def test_answer_only_evaluator_ignores_non_answer_positions():
    example = build_delayed_copy_example("ab", 3, distractor=".")
    vocabulary_size = len(example.vocabulary)
    probabilities = np.zeros((len(example.sequence), vocabulary_size), dtype=float)
    answer_rows = np.flatnonzero(example.answer_mask)
    targets = np.asarray(example.answer_targets)
    probabilities[answer_rows, (targets + 1) % vocabulary_size] = 0.05
    probabilities[answer_rows, targets] = 0.95
    probabilities[~example.answer_mask, 0] = 1.0

    loss, accuracy = answer_only_loss_and_accuracy(
        probabilities,
        example.answer_mask,
        example.answer_targets,
    )

    assert np.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0
    assert accuracy > 0.0


def test_model_evaluator_scores_next_character_rows(monkeypatch):
    example = build_delayed_copy_example("ij", 3, distractor=".")
    vocabulary = build_vocabulary("abcdefghij" + example.sequence)
    probabilities = np.zeros((len(example.sequence), vocabulary.size))
    answer_start = int(np.flatnonzero(example.answer_mask)[0])
    prediction_rows = np.arange(answer_start - 1, answer_start + len(example.answer_targets) - 1)
    shared_targets = encode(vocabulary, example.payload)
    probabilities[prediction_rows, shared_targets] = 1.0

    def controlled_forward_sequence(params, inputs, initial_hidden):
        return (), (), tuple(probabilities)

    monkeypatch.setattr(delayed_copy_eval, "forward_sequence", controlled_forward_sequence)
    _, accuracy, prediction = _evaluate_example("rnn", None, example, 3, vocabulary)

    assert accuracy == 1.0
    assert prediction[prediction_rows[0]:prediction_rows[-1] + 1] == example.payload
