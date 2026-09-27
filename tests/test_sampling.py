"""§ 'At test time ... we sample from this distribution, and feed it right
back in' and § 'Temperature'."""
import numpy as np

from domain.rnn_model import init_params
from domain.sampling import _temperature_scaled, sample
from domain.vocabulary import build_vocabulary


def test_sample_generates_the_requested_length_of_valid_indices():
    vocab = build_vocabulary("helo")
    params = init_params(vocab.size, hidden_size=8, seed=4)
    h0 = np.zeros((8, 1))

    generated = sample(
        params, vocab, h0, seed_index=0, length=20, temperature=1.0,
        rng=np.random.default_rng(0),
    )

    assert len(generated) == 20
    assert all(0 <= i < vocab.size for i in generated)


def test_sample_is_reproducible_given_the_same_rng_seed():
    vocab = build_vocabulary("helo")
    params = init_params(vocab.size, hidden_size=8, seed=4)
    h0 = np.zeros((8, 1))

    first = sample(params, vocab, h0, 0, 15, 1.0, np.random.default_rng(42))
    second = sample(params, vocab, h0, 0, 15, 1.0, np.random.default_rng(42))

    assert first == second


def test_temperature_near_zero_concentrates_almost_all_mass_on_the_favorite():
    """'setting temperature very near zero will give the most likely thing' --
    tested directly on the temperature-scaling function, since an untrained
    model's own logits can be too close together for this to show up after
    going through a full (stochastic) autoregressive sample -- see
    test_higher_temperature_gives_at_least_as_much_diversity_as_lower below
    for that composed, end-to-end version of the same idea."""
    p = np.array([[0.4], [0.3], [0.2], [0.1]])

    sharpened = _temperature_scaled(p, temperature=0.01)

    assert sharpened[0, 0] > 0.99


def test_temperature_above_one_flattens_the_distribution():
    """'higher temperatures will give more diversity' -- the favorite
    character's probability should shrink back towards the others."""
    p = np.array([[0.7], [0.1], [0.1], [0.1]])

    flattened = _temperature_scaled(p, temperature=5.0)

    assert flattened[0, 0] < p[0, 0]


def test_higher_temperature_gives_at_least_as_much_diversity_as_lower():
    vocab = build_vocabulary("the quick fox jumps over lazy dogs")
    params = init_params(vocab.size, hidden_size=16, seed=5)
    h0 = np.zeros((16, 1))

    low = sample(params, vocab, h0, 0, 60, temperature=0.01, rng=np.random.default_rng(2))
    high = sample(params, vocab, h0, 0, 60, temperature=2.0, rng=np.random.default_rng(2))

    assert len(set(high)) >= len(set(low))
