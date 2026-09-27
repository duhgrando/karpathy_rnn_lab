"""Infrastructure: deterministic delayed-copy task generation."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Sequence, Tuple

import numpy as np

from domain.vocabulary import build_vocabulary, encode

START_TOKEN = "\x02"
END_TOKEN = "\x03"


@dataclass(frozen=True)
class DelayedCopyExample:
    payload: str
    delay: int
    distractor: str
    sequence: str
    prompt: str
    answer_mask: np.ndarray
    answer_tokens: Tuple[str, ...]
    answer_targets: Tuple[int, ...]
    vocabulary: Tuple[str, ...]

    def __eq__(self, other):
        if not isinstance(other, DelayedCopyExample):
            return NotImplemented
        return (
            self.payload,
            self.delay,
            self.distractor,
            self.sequence,
            self.prompt,
            self.answer_mask.tobytes(),
            self.answer_tokens,
            self.answer_targets,
            self.vocabulary,
        ) == (
            other.payload,
            other.delay,
            other.distractor,
            other.sequence,
            other.prompt,
            other.answer_mask.tobytes(),
            other.answer_tokens,
            other.answer_targets,
            other.vocabulary,
        )


def build_delayed_copy_example(
    payload: str,
    delay: int,
    distractor: str = ".",
) -> DelayedCopyExample:
    if not payload:
        raise ValueError("payload must be non-empty")
    if delay < 0:
        raise ValueError("delay must be non-negative")
    if not distractor:
        raise ValueError("distractor must be non-empty")

    prompt = f"{START_TOKEN}!{payload}#"
    sequence = f"{prompt}{distractor * delay}={payload}{END_TOKEN}"
    vocab = build_vocabulary(sequence)
    answer_tokens = tuple(payload)
    answer_mask = np.zeros(len(sequence), dtype=bool)
    answer_start = len(sequence) - len(answer_tokens) - len(END_TOKEN)
    answer_mask[answer_start:answer_start + len(answer_tokens)] = True
    answer_targets = tuple(map(int, encode(vocab, payload)))
    return DelayedCopyExample(
        payload=payload,
        delay=delay,
        distractor=distractor,
        sequence=sequence,
        prompt=prompt,
        answer_mask=answer_mask,
        answer_tokens=answer_tokens,
        answer_targets=answer_targets,
        vocabulary=vocab.chars,
    )


def generate_delayed_copy_dataset(
    payloads: Sequence[str],
    delay: int,
    seed: int = 0,
) -> Tuple[Tuple[DelayedCopyExample, ...], Tuple[DelayedCopyExample, ...]]:
    if not payloads:
        raise ValueError("payloads must not be empty")

    rng = np.random.default_rng(seed)
    permuted = tuple(rng.permutation(len(payloads)))
    train_count = max(1, len(payloads) - 1)
    train_indices = permuted[:train_count]
    eval_indices = permuted[train_count:] or permuted[-1:]
    build_example = partial(build_delayed_copy_example, delay=delay)
    train_examples = tuple(map(build_example, map(payloads.__getitem__, train_indices)))
    eval_examples = tuple(map(build_example, map(payloads.__getitem__, eval_indices)))
    return train_examples, eval_examples
