"""Guard caller-owned inputs against mutation by domain/application functions."""
from __future__ import annotations

import copy
import importlib
import inspect
import pkgutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields, is_dataclass
from functools import singledispatch

import numpy as np
import pytest

import application
import domain
from application.train_service import (
    TrainerState,
    TrainingConfig,
    _make_batches,
    _run_batch,
    _run_epoch,
    train,
)
from domain.lstm_model import (
    compute_gates,
    initialize_lstm_parameters,
    lstm_forward_sequence,
    lstm_step,
    sigmoid,
)
from domain.lstm_training import LSTMMemory, lstm_backpropagate_through_time
from domain.optimization import adagrad_step, adagrad_update, clip_gradients, zero_memory
from domain.rnn_model import (
    cross_entropy_loss,
    forward_sequence,
    forward_step,
    initialize_rnn_parameters,
    softmax,
)
from domain.sampling import _temperature_scaled, sample
from domain.stacked_rnn import (
    _append_layer_states,
    initialize_stacked_rnn_parameters,
    stacked_forward_sequence,
    stacked_step,
)
from domain.stacked_training import (
    _accumulate_layer_gradients,
    _initialize_layer_gradients,
    clip_stacked_gradients,
    stacked_adagrad_update,
    stacked_backpropagate_through_time,
    zero_stacked_memory,
)
from domain.training import AdagradMemory, backpropagate_through_time
from domain.vocabulary import (
    build_vocabulary,
    char_to_index,
    decode,
    encode,
    index_to_char,
    one_hot,
)

@dataclass(frozen=True)
class FunctionSpec:
    name: str
    function: Callable[..., object]
    arguments: tuple[object, ...]
    keyword_arguments: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class MutationResult:
    name: str
    inputs_unchanged: bool


@dataclass(frozen=True)
class SequenceCase:
    vocabulary: object
    character_indices: tuple[int, ...]
    inputs: list[np.ndarray]
    targets: tuple[int, ...]


@dataclass(frozen=True)
class RNNCase:
    params: object
    initial_hidden: np.ndarray
    hidden_states: tuple
    probabilities: tuple
    gradients: object
    memory: object


@dataclass(frozen=True)
class LSTMCase:
    params: object
    initial_cell: np.ndarray
    hidden_states: tuple
    cell_states: tuple
    probabilities: tuple
    gradients: object
    memory: object


@dataclass(frozen=True)
class StackedCase:
    params: object
    initial_hidden_states: tuple[np.ndarray, ...]
    hidden_states: tuple
    probabilities: tuple
    gradients: object
    memory: object
    layer_gradients: list
    next_hidden_gradients: list


def _source_modules():
    return tuple(
        importlib.import_module(module_info.name)
        for package in (application, domain)
        for module_info in pkgutil.walk_packages(
            package.__path__, prefix=f"{package.__name__}."
        )
    )


_FUNCTION_MODULES = _source_modules()


@singledispatch
def _snapshot(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _snapshot_dataclass(value)
    return copy.deepcopy(value)


def _snapshot_dataclass(value):
    return (
        type(value),
        tuple((field.name, _snapshot(getattr(value, field.name))) for field in fields(value)),
    )


@_snapshot.register(np.ndarray)
def _snapshot_array(value):
    return ("array", value.dtype.str, value.shape, value.tobytes())


@_snapshot.register(np.random.Generator)
def _snapshot_generator(value):
    return ("generator", _snapshot(value.bit_generator.state))


@_snapshot.register(dict)
def _snapshot_dictionary(value):
    return ("dict", tuple((key, _snapshot(item)) for key, item in value.items()))


@_snapshot.register(list)
def _snapshot_list(value):
    return ("list", tuple(_snapshot(item) for item in value))


@_snapshot.register(tuple)
def _snapshot_tuple(value):
    return ("tuple", tuple(_snapshot(item) for item in value))


def _scratch_gradient_inputs(params, initial_hidden_states):
    layer_gradients = [
        {
            "dWxh": np.zeros_like(layer.Wxh),
            "dWhh": np.zeros_like(layer.Whh),
            "dbh": np.zeros_like(layer.bh),
        }
        for layer in params.layers
    ]
    hidden_gradients = [np.zeros_like(state) for state in initial_hidden_states]
    return layer_gradients, hidden_gradients


def _run_spec(spec):
    arguments, keyword_arguments = copy.deepcopy(
        (spec.arguments, dict(spec.keyword_arguments))
    )
    before = _snapshot((arguments, keyword_arguments))
    result = spec.function(*arguments, **keyword_arguments)
    if isinstance(result, Iterator):
        tuple(result)
    unchanged = before == _snapshot((arguments, keyword_arguments))
    return MutationResult(spec.name, unchanged)


def _run_specs(specs):
    return tuple(_run_spec(spec) for spec in specs)


def _sequence_case():
    vocabulary = build_vocabulary("abba")
    character_indices = encode(vocabulary, "abba")
    inputs = [one_hot(vocabulary, index) for index in character_indices[:-1]]
    return SequenceCase(vocabulary, character_indices, inputs, character_indices[1:])


def _rnn_case(sequence):
    rnn_params = initialize_rnn_parameters(sequence.vocabulary.size, hidden_size=3, seed=1)
    initial_hidden = np.zeros((3, 1))
    rnn_hidden_states, _, probabilities = forward_sequence(
        rnn_params, sequence.inputs, initial_hidden
    )
    rnn_gradients, _ = backpropagate_through_time(
        rnn_params, sequence.inputs, sequence.targets, rnn_hidden_states, probabilities
    )
    rnn_memory = zero_memory(rnn_params, AdagradMemory)
    return RNNCase(rnn_params, initial_hidden, rnn_hidden_states, probabilities, rnn_gradients, rnn_memory)


def _lstm_case(sequence, rnn):
    lstm_params = initialize_lstm_parameters(sequence.vocabulary.size, hidden_size=3, seed=2)
    initial_cell = np.zeros((3, 1))
    lstm_hidden_states, lstm_cell_states, _, lstm_probabilities = lstm_forward_sequence(
        lstm_params, sequence.inputs, rnn.initial_hidden, initial_cell
    )
    lstm_gradients, _, _ = lstm_backpropagate_through_time(
        lstm_params, sequence.inputs, sequence.targets, lstm_hidden_states,
        lstm_cell_states, lstm_probabilities,
    )
    lstm_memory = zero_memory(lstm_params, LSTMMemory)
    return LSTMCase(
        lstm_params, initial_cell, lstm_hidden_states, lstm_cell_states,
        lstm_probabilities, lstm_gradients, lstm_memory,
    )


def _stacked_case(sequence):
    params = initialize_stacked_rnn_parameters(
        sequence.vocabulary.size, hidden_sizes=(3, 2), seed=3
    )
    initial_hidden_states = (np.zeros((3, 1)), np.zeros((2, 1)))
    hidden_states, _, probabilities = stacked_forward_sequence(
        params, sequence.inputs, initial_hidden_states
    )
    gradients = stacked_backpropagate_through_time(
        params, sequence.inputs, sequence.targets, hidden_states, probabilities
    )
    memory = zero_stacked_memory(params)
    scratch_layer_gradients, scratch_hidden_gradients = _scratch_gradient_inputs(
        params, initial_hidden_states
    )
    return StackedCase(
        params, initial_hidden_states, hidden_states, probabilities, gradients, memory,
        scratch_layer_gradients, scratch_hidden_gradients,
    )


def _spec(name, function, *arguments):
    return FunctionSpec(name, function, arguments, ())


def _vocabulary_specs(sequence):
    vocabulary = sequence.vocabulary
    indices = sequence.character_indices
    return (
        _spec("vocabulary.build", build_vocabulary, "abba"),
        _spec("vocabulary.char_to_index", char_to_index, vocabulary, "a"),
        _spec("vocabulary.index_to_char", index_to_char, vocabulary, indices[0]),
        _spec("vocabulary.encode", encode, vocabulary, "abba"),
        _spec("vocabulary.decode", decode, vocabulary, indices),
        _spec("vocabulary.one_hot", one_hot, vocabulary, indices[0]),
    )


def _rnn_specs(sequence, case):
    return (
        _spec("rnn.initialize_parameters", initialize_rnn_parameters, sequence.vocabulary.size, 3, 6),
        _spec("rnn.softmax", softmax, np.array([[1.0], [2.0]])),
        _spec("rnn.forward_step", forward_step, case.params, sequence.inputs[0], case.initial_hidden),
        _spec("rnn.forward_sequence", forward_sequence, case.params, sequence.inputs, case.initial_hidden),
        _spec("rnn.cross_entropy_loss", cross_entropy_loss, case.probabilities, sequence.targets),
        _spec(
            "rnn.backpropagate_through_time", backpropagate_through_time, case.params,
            sequence.inputs, sequence.targets, case.hidden_states, case.probabilities,
        ),
    )


def _lstm_specs(sequence, rnn, case):
    return (
        _spec("lstm.initialize_parameters", initialize_lstm_parameters, sequence.vocabulary.size, 3, 7),
        _spec("lstm.sigmoid", sigmoid, np.array([[0.5], [-0.5]])),
        _spec("lstm.compute_gates", compute_gates, case.params, sequence.inputs[0], rnn.initial_hidden),
        _spec("lstm.step", lstm_step, case.params, sequence.inputs[0], rnn.initial_hidden, case.initial_cell),
        _spec(
            "lstm.forward_sequence", lstm_forward_sequence, case.params, sequence.inputs,
            rnn.initial_hidden, case.initial_cell,
        ),
        _spec(
            "lstm.backpropagate_through_time", lstm_backpropagate_through_time, case.params,
            sequence.inputs, sequence.targets, case.hidden_states, case.cell_states,
            case.probabilities,
        ),
    )


def _optimization_specs(rnn, lstm):
    return (
        _spec("optimization.clip_gradients", clip_gradients, rnn.gradients),
        _spec("optimization.zero_memory", zero_memory, rnn.params, AdagradMemory),
        _spec("optimization.adagrad_step", adagrad_step, rnn.params.Wxh, rnn.gradients.dWxh, rnn.memory.mWxh, 0.1),
        _spec("optimization.adagrad_update", adagrad_update, rnn.params, rnn.gradients, rnn.memory),
        _spec("optimization.clip_lstm_gradients", clip_gradients, lstm.gradients),
        _spec("optimization.zero_lstm_memory", zero_memory, lstm.params, LSTMMemory),
        _spec("optimization.adagrad_lstm_update", adagrad_update, lstm.params, lstm.gradients, lstm.memory),
    )


def _sampling_specs(sequence, rnn):
    return (
        _spec("sampling.temperature_scaled", _temperature_scaled, rnn.probabilities[0], 0.5),
        _spec(
            "sampling.sample", sample, rnn.params, sequence.vocabulary, rnn.initial_hidden,
            sequence.character_indices[0], 4, 1.0, 5,
        ),
    )


def _stacked_specs(sequence, case):
    return (
        _spec("stacked.step", stacked_step, case.params, sequence.inputs[0], case.initial_hidden_states),
        _spec(
            "stacked.initialize_parameters", initialize_stacked_rnn_parameters,
            sequence.vocabulary.size, [3, 2], 8,
        ),
        _spec(
            "stacked.append_layer_states", _append_layer_states,
            [[state.copy()] for state in case.initial_hidden_states], case.initial_hidden_states,
        ),
        _spec("stacked.forward_sequence", stacked_forward_sequence, case.params, sequence.inputs, case.initial_hidden_states),
    )


def _stacked_training_specs(sequence, case):
    return (
        _spec("stacked_training.zero_memory", zero_stacked_memory, case.params),
        _spec("stacked_training.clip_gradients", clip_stacked_gradients, case.gradients),
        _spec("stacked_training.adagrad_update", stacked_adagrad_update, case.params, case.gradients, case.memory),
        _spec("stacked_training.initialize_layer_gradients", _initialize_layer_gradients, case.params, case.hidden_states),
        _spec(
            "stacked_training.accumulate_layer_gradients", _accumulate_layer_gradients,
            case.params, sequence.inputs, case.hidden_states, 0,
            np.ones_like(case.params.Why[:1].T), case.layer_gradients, case.next_hidden_gradients,
        ),
        _spec(
            "stacked_training.backpropagate_through_time", stacked_backpropagate_through_time,
            case.params, sequence.inputs, sequence.targets, case.hidden_states, case.probabilities,
        ),
    )


def _application_specs(sequence, rnn):
    config = TrainingConfig(hidden_size=3, seq_length=2, seed=4)
    trainer_state = TrainerState(
        params=rnn.params, memory=rnn.memory, hidden=rnn.initial_hidden.copy(),
        smooth_loss=1.0, iteration=0,
    )
    indices = sequence.character_indices
    return (
        _spec("application.make_batches", _make_batches, indices, 2),
        _spec("application.run_batch", _run_batch, sequence.vocabulary, config, (indices[:2], indices[1:3]), trainer_state),
        _spec("application.run_epoch", _run_epoch, sequence.vocabulary, config, indices, trainer_state),
        _spec("application.train", train, "abba", sequence.vocabulary, config, 1),
    )


def _function_specs():
    sequence = _sequence_case()
    rnn = _rnn_case(sequence)
    lstm = _lstm_case(sequence, rnn)
    stacked = _stacked_case(sequence)
    return (
        *_vocabulary_specs(sequence),
        *_rnn_specs(sequence, rnn),
        *_lstm_specs(sequence, rnn, lstm),
        *_optimization_specs(rnn, lstm),
        *_sampling_specs(sequence, rnn),
        *_stacked_specs(sequence, stacked),
        *_stacked_training_specs(sequence, stacked),
        *_application_specs(sequence, rnn),
    )


FUNCTION_SPECS = _function_specs()


@pytest.mark.parametrize("spec", FUNCTION_SPECS, ids=lambda spec: spec.name)
def test_function_does_not_mutate_its_arguments(spec):
    result, = _run_specs((spec,))
    assert result.inputs_unchanged, f"{result.name} mutated one or more input objects"


def _module_functions(module):
    return {
        function
        for function in vars(module).values()
        if inspect.isfunction(function) and function.__module__ == module.__name__
    }


def _missing_function_cases():
    defined_functions = set().union(*(_module_functions(module) for module in _FUNCTION_MODULES))
    tested_functions = {spec.function for spec in FUNCTION_SPECS}
    return sorted(
        f"{function.__module__}.{function.__name__}"
        for function in defined_functions - tested_functions
    )


def test_every_domain_and_application_function_has_a_mutation_case():
    missing_cases = _missing_function_cases()
    assert not missing_cases, f"Add mutation cases for: {', '.join(missing_cases)}"