"""Gradient clipping isn't named explicitly in the article's prose, but it is
part of the reference min-char-rnn implementation the article links to, and
of the BPTT/Adagrad machinery this project mirrors -- it keeps a single bad
batch from throwing the parameters somewhere the tanh nonlinearity can't
recover from."""
import numpy as np

from domain.optimization import clip_gradients
from domain.training import Gradients


def _extreme_gradients():
    return Gradients(
        dWxh=np.array([[100.0, -100.0]]),
        dWhh=np.array([[0.0]]),
        dWhy=np.array([[3.0]]),
        dbh=np.array([[0.0]]),
        dby=np.array([[0.0]]),
    )


def test_clip_gradients_bounds_large_values_to_the_clip_range():
    clipped = clip_gradients(_extreme_gradients(), clip=5.0)

    assert np.all(clipped.dWxh <= 5.0)
    assert np.all(clipped.dWxh >= -5.0)


def test_clip_gradients_leaves_small_values_untouched():
    clipped = clip_gradients(_extreme_gradients(), clip=5.0)

    assert clipped.dWhy[0, 0] == 3.0
