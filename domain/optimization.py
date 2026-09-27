"""Domain: optimizer mechanics shared by every cell type in this project.

Gradient clipping and the Adagrad update don't actually care whether the
gradients came from a vanilla RNN or an LSTM -- they only care that a
Gradients dataclass's fields are named `d<Name>`, a Params dataclass's
fields are named `<Name>`, and a Memory dataclass's fields are named
`m<Name>`, all lining up 1:1. That's a convention every cell type in this
project follows (see domain/training.py's Gradients/AdagradMemory and
domain/lstm_training.py's LSTMGradients/LSTMMemory), so these functions are
written once here, structurally, via `dataclasses.fields`, instead of once
per cell type.

The per-cell-type piece that genuinely differs is BPTT itself (the chain
rule through a tanh cell vs. four gates is different math) -- that stays
next to each cell's forward pass, in training.py / lstm_training.py.
"""
from __future__ import annotations

import dataclasses
from typing import Tuple, Type, TypeVar

import numpy as np

Grads = TypeVar("Grads")
Params = TypeVar("Params")
Memory = TypeVar("Memory")


def clip_gradients(grads: Grads, clip: float = 5.0) -> Grads:
    """Mitigate exploding gradients, as the reference implementation does."""
    clipped_fields = {
        field.name: np.clip(getattr(grads, field.name), -clip, clip)
        for field in dataclasses.fields(grads)
    }
    return type(grads)(**clipped_fields)


def zero_memory(params: Params, memory_cls: Type[Memory]) -> Memory:
    """Build a zero Adagrad memory matching `params`'s shapes: a params
    field `Wxh` gets a corresponding memory field `mWxh`."""
    memory_fields = {
        f"m{field.name}": np.zeros_like(getattr(params, field.name))
        for field in dataclasses.fields(params)
    }
    return memory_cls(**memory_fields)


def adagrad_step(
    param: np.ndarray, grad: np.ndarray, mem: np.ndarray, learning_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """The raw per-array Adagrad update, exposed as a public building block:
    adagrad_update() below uses it via the naming-convention reflection, and
    domain/stacked_training.py uses it directly for the same purpose on a
    nested (multi-layer) parameter structure that convention doesn't reach."""
    eps = 1e-8
    new_mem = mem + grad * grad
    new_param = param - learning_rate * grad / np.sqrt(new_mem + eps)
    return new_param, new_mem


def adagrad_update(
    params: Params, grads: Grads, memory: Memory, learning_rate: float = 0.1
) -> Tuple[Params, Memory]:
    """Per-parameter adaptive learning rate update, generic over any
    (Params, Gradients, Memory) triple that follows the naming convention
    described in the module docstring."""
    new_param_fields = {}
    new_memory_fields = {}
    for grad_field in dataclasses.fields(grads):
        name = grad_field.name[1:]  # "dWxh" -> "Wxh"
        new_param, new_mem = adagrad_step(
            getattr(params, name),
            getattr(grads, grad_field.name),
            getattr(memory, f"m{name}"),
            learning_rate,
        )
        new_param_fields[name] = new_param
        new_memory_fields[f"m{name}"] = new_mem
    return type(params)(**new_param_fields), type(memory)(**new_memory_fields)
