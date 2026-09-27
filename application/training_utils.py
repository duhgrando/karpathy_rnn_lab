"""Small shared helpers for character-model training services."""
from __future__ import annotations

from typing import Iterator, Tuple


def make_character_batches(
    indices: Tuple[int, ...], seq_length: int
) -> Iterator[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Yield fixed-length input windows and their one-character-shifted targets."""
    last_start = len(indices) - seq_length - 1
    for start in range(0, max(last_start, 0), seq_length):
        yield indices[start:start + seq_length], indices[start + 1:start + seq_length + 1]