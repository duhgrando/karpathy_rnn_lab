"""domain/optimization.py's whole reason to exist: clip_gradients,
zero_memory, and adagrad_update are written once, structurally, and work
for *any* Params/Gradients/Memory triple that follows this project's
`<Name>` / `d<Name>` / `m<Name>` naming convention -- not just the vanilla
RNN's. These tests exercise the same three functions against both the RNN's
dataclasses and the LSTM's, to prove that claim rather than just assert it
in a docstring.
"""
import numpy as np

from domain.lstm_model import init_lstm_params
from domain.lstm_training import LSTMMemory
from domain.optimization import adagrad_update, clip_gradients, zero_memory
from domain.rnn_model import init_params
from domain.training import AdagradMemory, Gradients
from domain.lstm_training import LSTMGradients


def test_zero_memory_works_for_both_rnn_and_lstm_params():
    rnn_params = init_params(vocab_size=4, hidden_size=3, seed=0)
    lstm_params = init_lstm_params(vocab_size=4, hidden_size=3, seed=0)

    rnn_memory = zero_memory(rnn_params, AdagradMemory)
    lstm_memory = zero_memory(lstm_params, LSTMMemory)

    assert isinstance(rnn_memory, AdagradMemory)
    assert isinstance(lstm_memory, LSTMMemory)
    assert np.array_equal(rnn_memory.mWxh, np.zeros_like(rnn_params.Wxh))
    assert np.array_equal(lstm_memory.mWxi, np.zeros_like(lstm_params.Wxi))


def test_clip_gradients_works_for_both_gradient_shapes():
    rnn_grads = Gradients(
        dWxh=np.array([[100.0]]), dWhh=np.array([[0.0]]), dWhy=np.array([[0.0]]),
        dbh=np.array([[0.0]]), dby=np.array([[0.0]]),
    )
    lstm_grads = LSTMGradients(
        dWxi=np.array([[100.0]]), dWhi=np.array([[0.0]]), dbi=np.array([[0.0]]),
        dWxf=np.array([[0.0]]), dWhf=np.array([[0.0]]), dbf=np.array([[0.0]]),
        dWxo=np.array([[0.0]]), dWho=np.array([[0.0]]), dbo=np.array([[0.0]]),
        dWxg=np.array([[0.0]]), dWhg=np.array([[0.0]]), dbg=np.array([[0.0]]),
        dWhy=np.array([[0.0]]), dby=np.array([[0.0]]),
    )

    assert clip_gradients(rnn_grads, clip=5.0).dWxh[0, 0] == 5.0
    assert clip_gradients(lstm_grads, clip=5.0).dWxi[0, 0] == 5.0


def test_adagrad_update_moves_both_kinds_of_params_against_their_gradient():
    rnn_params = init_params(vocab_size=4, hidden_size=3, seed=1)
    lstm_params = init_lstm_params(vocab_size=4, hidden_size=3, seed=1)
    rnn_memory = zero_memory(rnn_params, AdagradMemory)
    lstm_memory = zero_memory(lstm_params, LSTMMemory)

    rnn_grads = Gradients(**{
        f.name: np.ones_like(getattr(rnn_params, f.name[1:]))
        for f in Gradients.__dataclass_fields__.values()
    })
    lstm_grads = LSTMGradients(**{
        f.name: np.ones_like(getattr(lstm_params, f.name[1:]))
        for f in LSTMGradients.__dataclass_fields__.values()
    })

    new_rnn_params, _ = adagrad_update(rnn_params, rnn_grads, rnn_memory, learning_rate=0.5)
    new_lstm_params, _ = adagrad_update(lstm_params, lstm_grads, lstm_memory, learning_rate=0.5)

    assert np.all(new_rnn_params.Wxh < rnn_params.Wxh)
    assert np.all(new_lstm_params.Wxi < lstm_params.Wxi)
