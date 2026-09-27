"""§ '(per-parameter adaptive learning rate methods) to stabilize the
updates' -- Adagrad, as used in Karpathy's reference implementation."""
import numpy as np

from domain.optimization import adagrad_update, zero_memory
from domain.rnn_model import init_params
from domain.training import AdagradMemory, Gradients


def _ones_gradients(params):
    return Gradients(
        dWxh=np.ones_like(params.Wxh),
        dWhh=np.ones_like(params.Whh),
        dWhy=np.ones_like(params.Why),
        dbh=np.ones_like(params.bh),
        dby=np.ones_like(params.by),
    )


def test_adagrad_moves_parameters_against_the_gradient():
    params = init_params(vocab_size=4, hidden_size=3, seed=2)
    memory = zero_memory(params, AdagradMemory)
    grads = _ones_gradients(params)

    new_params, new_memory = adagrad_update(params, grads, memory, learning_rate=0.5)

    assert np.all(new_params.Wxh < params.Wxh)  # positive grad -> parameter decreases
    assert np.all(new_memory.mWxh > memory.mWxh)  # squared-grad memory accumulates


def test_adagrad_update_never_mutates_its_inputs():
    params = init_params(vocab_size=4, hidden_size=3, seed=3)
    memory = zero_memory(params, AdagradMemory)
    original_Wxh, original_memory = params.Wxh.copy(), memory.mWxh.copy()
    grads = _ones_gradients(params)

    adagrad_update(params, grads, memory, learning_rate=0.5)

    assert np.array_equal(params.Wxh, original_Wxh)
    assert np.array_equal(memory.mWxh, original_memory)


def test_larger_accumulated_memory_shrinks_the_effective_step():
    """The whole point of Adagrad: a parameter that has already seen large
    gradients gets a smaller effective learning rate than a fresh one."""
    params = init_params(vocab_size=4, hidden_size=3, seed=4)
    grads = _ones_gradients(params)
    fresh_memory = zero_memory(params, AdagradMemory)
    warmed_memory = AdagradMemory(
        mWxh=np.full_like(params.Wxh, 100.0),
        mWhh=np.zeros_like(params.Whh),
        mWhy=np.zeros_like(params.Why),
        mbh=np.zeros_like(params.bh),
        mby=np.zeros_like(params.by),
    )

    fresh_step_params, _ = adagrad_update(params, grads, fresh_memory, learning_rate=0.5)
    warmed_step_params, _ = adagrad_update(params, grads, warmed_memory, learning_rate=0.5)

    fresh_step_size = np.abs(fresh_step_params.Wxh - params.Wxh)
    warmed_step_size = np.abs(warmed_step_params.Wxh - params.Wxh)
    assert np.all(warmed_step_size < fresh_step_size)
