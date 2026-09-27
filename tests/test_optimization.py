"""domain/optimization.py's whole reason to exist: clip_gradients,
zero_memory, and adagrad_update are written once, structurally, and work
for *any* Params/Gradients/Memory triple that follows this project's
`<Name>` / `d<Name>` / `m<Name>` naming convention -- not just the vanilla
RNN's. These tests exercise the same three functions against both the RNN's
dataclasses and the LSTM's, to prove that claim rather than just assert it
in a docstring.
"""
import numpy as np

from domain.lstm_model import initialize_lstm_parameters
from domain.lstm_training import LSTMMemory
from domain.optimization import adagrad_update, clip_gradients, zero_memory
from domain.rnn_model import initialize_rnn_parameters
from domain.training import AdagradMemory, Gradients
from domain.lstm_training import LSTMGradients


def test_zero_memory_matches_rnn_parameter_shapes():
    rnn_params = initialize_rnn_parameters(vocab_size=4, hidden_size=3, seed=0)
    rnn_memory = zero_memory(rnn_params, AdagradMemory)

    assert isinstance(rnn_memory, AdagradMemory)
    assert np.array_equal(rnn_memory.mWxh, np.zeros_like(rnn_params.Wxh))


def test_zero_memory_matches_lstm_parameter_shapes():
    lstm_params = initialize_lstm_parameters(vocab_size=4, hidden_size=3, seed=0)
    lstm_memory = zero_memory(lstm_params, LSTMMemory)

    assert isinstance(lstm_memory, LSTMMemory)
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


def _ones_like_gradients(params, gradients_type):
    return gradients_type(**{
        field.name: np.ones_like(getattr(params, field.name[1:]))
        for field in gradients_type.__dataclass_fields__.values()
    })


def test_adagrad_update_moves_rnn_params_against_their_gradient():
    rnn_params = initialize_rnn_parameters(vocab_size=4, hidden_size=3, seed=1)
    rnn_memory = zero_memory(rnn_params, AdagradMemory)
    rnn_grads = _ones_like_gradients(rnn_params, Gradients)

    new_rnn_params, _ = adagrad_update(rnn_params, rnn_grads, rnn_memory, learning_rate=0.5)

    assert np.all(new_rnn_params.Wxh < rnn_params.Wxh)


def test_adagrad_update_moves_lstm_params_against_their_gradient():
    lstm_params = initialize_lstm_parameters(vocab_size=4, hidden_size=3, seed=1)
    lstm_memory = zero_memory(lstm_params, LSTMMemory)
    lstm_grads = _ones_like_gradients(lstm_params, LSTMGradients)

    new_lstm_params, _ = adagrad_update(lstm_params, lstm_grads, lstm_memory, learning_rate=0.5)

    assert np.all(new_lstm_params.Wxi < lstm_params.Wxi)
