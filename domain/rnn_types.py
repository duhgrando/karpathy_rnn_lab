"""Semantic array types shared by character encoding and RNN computation."""
from typing import NewType, TypeAlias

import numpy as np

RNNArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]
RNNInput = NewType("RNNInput", RNNArray)
HiddenState = NewType("HiddenState", RNNArray)
LogitOutput = NewType("LogitOutput", RNNArray)
ProbabilityOutput = NewType("ProbabilityOutput", RNNArray)
