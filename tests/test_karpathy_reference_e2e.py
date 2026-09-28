"""Compare the project trainer with Karpathy's minimal character RNN.

Reference: https://gist.github.com/karpathy/d4dee566867f8291f086
The test oracle is a Python 3 adaptation using identical initial parameters
and vocabulary order for a deterministic comparison.
"""
from pathlib import Path

import numpy as np

from application.train_service import TrainingConfig, TrainingResult, train
from domain.rnn_model import (
    HiddenState,
    ProbabilityOutput,
    RNNOutput,
    RNNParams,
    forward_sequence,
    forward_step,
)
from domain.vocabulary import Vocabulary, build_vocabulary, encode, one_hot
from domain.rnn_types import RNNInput
from infrastructure.corpus import load_corpus
from tests.karpathy_reference_adapter import (
    PARAMETER_NAMES,
    KarpathyReferenceAdapter,
    _apply_adagrad,
)


def _assert_parameters_match(actual: RNNParams, expected: RNNParams) -> None:
    """Check every model parameter against the reference within float tolerance."""
    for name in PARAMETER_NAMES:
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=1e-10, atol=1e-12
        )


def _assert_training_matches(actual: TrainingResult, expected: TrainingResult) -> None:
    """Compare batch counts, losses, and parameter snapshots from both trainers."""
    assert len(actual.snapshots) == len(expected.snapshots) == 4
    for actual_snapshot, expected_snapshot in zip(actual.snapshots, expected.snapshots):
        assert actual_snapshot.iteration == expected_snapshot.iteration
        np.testing.assert_allclose(
            actual_snapshot.smooth_loss, expected_snapshot.smooth_loss, rtol=1e-11
        )
        _assert_parameters_match(actual_snapshot.params, expected_snapshot.params)
    _assert_parameters_match(actual.params, expected.params)


def _assert_step_outputs_match(
    actual: RNNOutput, expected: RNNOutput
) -> None:
    """Compare named hidden, logit, and probability values from one step."""
    np.testing.assert_allclose(
        actual.hidden_state, expected.hidden_state, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(actual.logits, expected.logits, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        actual.probabilities, expected.probabilities, rtol=1e-10, atol=1e-12
    )


def _assert_array_maps_equal(
    actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]
) -> None:
    for name in PARAMETER_NAMES:
        np.testing.assert_array_equal(actual[name], expected[name])


def _assert_array_maps_do_not_share_memory(
    updated: dict[str, np.ndarray], original: dict[str, np.ndarray]
) -> None:
    assert all(
        not np.shares_memory(updated[name], original[name])
        for name in PARAMETER_NAMES
    )


def _make_adagrad_arrays(fill_value: float) -> dict[str, np.ndarray]:
    """Create same-shaped parameter, gradient, or memory arrays."""
    return {name: np.full((2, 2), fill_value) for name in PARAMETER_NAMES}


def _copy_array_map(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Snapshot every array in a parameter-shaped mapping."""
    return {name: value.copy() for name, value in arrays.items()}


def test_reference_adagrad_does_not_mutate_its_inputs() -> None:
    """Return updated arrays while leaving params, gradients, and memory intact."""
    params: dict[str, np.ndarray] = _make_adagrad_arrays(0.5)
    gradients: dict[str, np.ndarray] = _make_adagrad_arrays(1.0)
    memory: dict[str, np.ndarray] = _make_adagrad_arrays(0.0)
    original_params: dict[str, np.ndarray] = _copy_array_map(params)
    original_gradients: dict[str, np.ndarray] = _copy_array_map(gradients)
    original_memory: dict[str, np.ndarray] = _copy_array_map(memory)

    updated_params, updated_memory = _apply_adagrad(
        params, gradients, memory, learning_rate=0.1
    )

    _assert_array_maps_equal(params, original_params)
    _assert_array_maps_equal(gradients, original_gradients)
    _assert_array_maps_equal(memory, original_memory)
    _assert_array_maps_do_not_share_memory(updated_params, params)
    _assert_array_maps_do_not_share_memory(updated_memory, memory)


def test_training_matches_karpathy_reference_on_tiny_shakespeare() -> None:
    """Compare one epoch of training and fixed-input predictions on Shakespeare."""
    corpus_path: Path = (
        Path(__file__).parents[1]
        / "infrastructure"
        / "input_tinyshakespeare.txt"
    )
    corpus: str = load_corpus(str(corpus_path))[:102]
    vocab: Vocabulary = build_vocabulary(corpus)
    config: TrainingConfig = TrainingConfig(
        hidden_size=8, seq_length=25, learning_rate=0.1, seed=0
    )

    initial_params: RNNParams = train(corpus, vocab, config, epochs=0).params
    actual: TrainingResult = train(corpus, vocab, config, epochs=1)
    reference: KarpathyReferenceAdapter = KarpathyReferenceAdapter(initial_params)
    expected: TrainingResult = reference.train(corpus, vocab, config, epochs=1)
    _assert_training_matches(actual, expected)

    input_ids: tuple[int, ...] = encode(vocab, corpus[80:88])
    inputs: list[RNNInput] = [
        one_hot(vocab, index) for index in input_ids
    ]
    h0: HiddenState = HiddenState(np.zeros((config.hidden_size, 1)))
    actual_step: RNNOutput = forward_step(actual.params, inputs[0], h0)
    expected_step: RNNOutput = reference.forward_step(
        expected.params, inputs[0], h0
    )
    _assert_step_outputs_match(actual_step, expected_step)

    actual_probabilities: tuple[ProbabilityOutput, ...] = forward_sequence(
        actual.params, inputs, h0
    )[2]
    expected_probabilities: tuple[ProbabilityOutput, ...] = reference.next_character_probabilities(
        expected.params, vocab, input_ids
    )
    for actual_probability, expected_probability in zip(
        actual_probabilities, expected_probabilities
    ):
        np.testing.assert_allclose(
            actual_probability, expected_probability, rtol=1e-10, atol=1e-12
        )
