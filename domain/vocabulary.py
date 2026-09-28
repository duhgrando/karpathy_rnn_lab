"""Domain: character vocabulary.

Maps directly to the "Character-Level Language Models" section of the
article: a vocabulary of the distinct characters in a corpus, and the
1-of-k ("one hot") encoding used to feed characters into the RNN.

Everything here is a pure function of its arguments; a Vocabulary is an
immutable value object, not something callers mutate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from domain.rnn_types import RNNInput


@dataclass(frozen=True)
class Vocabulary:
    chars: Tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.chars)


def build_vocabulary(corpus: str) -> Vocabulary:
    """Derive a stable, sorted vocabulary from a corpus."""
    return Vocabulary(chars=tuple(sorted(set(corpus))))


def char_to_index(vocab: Vocabulary, char: str) -> int:
    return vocab.chars.index(char)


def index_to_char(vocab: Vocabulary, index: int) -> str:
    return vocab.chars[index]


def encode(vocab: Vocabulary, text: str) -> Tuple[int, ...]:
    """Map a string to a tuple of vocabulary indices."""
    return tuple(char_to_index(vocab, c) for c in text)


def decode(vocab: Vocabulary, indices: Tuple[int, ...]) -> str:
    return "".join(index_to_char(vocab, i) for i in indices)


def one_hot(vocab: Vocabulary, index: int) -> RNNInput:
    """1-of-k encoding: all zero except a single one at the character's index."""
    vector = np.zeros((vocab.size, 1))
    vector[index, 0] = 1.0
    return RNNInput(vector)
