"""Infrastructure: where training text comes from.

Kept separate from the domain so the RNN/BPTT/sampling code never knows or
cares whether the text came from a file, a URL, or this fallback string.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# A short, deliberately repetitive, originally-written corpus. It exists so
# the end-to-end test can finish in well under a second while still giving a
# checkable version of the article's "unreasonable effectiveness" claim.
# Point load_corpus() at a real text file (a public-domain novel, your own
# notes, source code, ...) to reproduce the article's own experiments.
DEFAULT_CORPUS = "the quick fox runs. the quick fox jumps. the quick fox hides. " * 20


def load_corpus(path: Optional[str] = None) -> str:
    if path is None:
        return DEFAULT_CORPUS
    return Path(path).read_text(encoding="utf-8")
