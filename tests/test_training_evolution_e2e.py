"""§ 'The evolution of samples while training' and the article's central
claim: a very simple model, given enough iterations, becomes surprisingly
effective at reproducing structure in its own training text.

This end-to-end test is the executable analogue of Karpathy's War-and-Peace
progression (random jumbles -> word-like chunks -> real words): it trains a
tiny RNN on a short repetitive corpus and checks (1) that loss falls
substantially over training, and (2) that a low-temperature sample from the
trained model reproduces whole words from the training text -- the toy-scale
version of the "is that they were all the same thing that was a startup"
near-memorization example the article shows under low temperature.
"""
from pathlib import Path

import numpy as np

from application.train_service import TrainingConfig, sample_from_model, train
from domain.rnn_model import cross_entropy_loss, forward_sequence
from domain.vocabulary import build_vocabulary, encode, one_hot
from infrastructure.corpus import load_corpus


def test_loss_falls_substantially_over_training():
    """Compares raw cross-entropy loss on the same window of text before and
    after training -- deliberately *not* train()'s internal EMA smooth_loss,
    which (like Karpathy's reference implementation) decays too slowly
    (0.999) to move much within a single short test run."""
    corpus = load_corpus()
    vocab = build_vocabulary(corpus)
    config = TrainingConfig(hidden_size=32, seq_length=20, learning_rate=0.1, seed=0)
    indices = encode(vocab, corpus)
    window_inputs = [one_hot(vocab, i) for i in indices[:config.seq_length]]
    window_targets = indices[1:config.seq_length + 1]
    h0 = np.zeros((config.hidden_size, 1))

    initial_params = train(corpus, vocab, config, epochs=0).params  # untrained baseline
    trained_params = train(corpus, vocab, config, epochs=20).params

    _, _, initial_ps = forward_sequence(initial_params, window_inputs, h0)
    _, _, trained_ps = forward_sequence(trained_params, window_inputs, h0)
    initial_loss = cross_entropy_loss(initial_ps, window_targets)
    trained_loss = cross_entropy_loss(trained_ps, window_targets)

    assert trained_loss < initial_loss * 0.2


def test_tiny_shakespeare_file_trains_and_samples_end_to_end():
    corpus_path = Path(__file__).parents[1] / "infrastructure" / "input_tinyshakespeare.txt"
    corpus = load_corpus(str(corpus_path))[:2000]
    vocab = build_vocabulary(corpus)
    config = TrainingConfig(hidden_size=8, seq_length=20, learning_rate=0.1, seed=0)
    indices = encode(vocab, corpus)
    window_inputs = [one_hot(vocab, index) for index in indices[:config.seq_length]]
    window_targets = indices[1:config.seq_length + 1]
    initial_params = train(corpus, vocab, config, epochs=0).params
    trained = train(corpus, vocab, config, epochs=4)
    h0 = np.zeros((config.hidden_size, 1))

    _, _, initial_ps = forward_sequence(initial_params, window_inputs, h0)
    _, _, trained_ps = forward_sequence(trained.params, window_inputs, h0)
    initial_loss = cross_entropy_loss(initial_ps, window_targets)
    trained_loss = cross_entropy_loss(trained_ps, window_targets)
    generated = sample_from_model(
        trained.params,
        vocab,
        h0,
        seed_index=indices[0],
        length=40,
        temperature=0.8,
        seed=0,
    )

    assert trained_loss < initial_loss
    assert len(generated) == 40


def test_low_temperature_sample_after_training_reproduces_training_words():
    corpus = load_corpus()
    vocab = build_vocabulary(corpus)
    config = TrainingConfig(hidden_size=32, seq_length=20, learning_rate=0.1, seed=0)

    params = train(corpus, vocab, config, epochs=8).params

    seed_index = encode(vocab, "t")[0]
    h0 = np.zeros((config.hidden_size, 1))
    generated = sample_from_model(
        params, vocab, h0, seed_index=seed_index, length=40,
        temperature=0.2, seed=0,
    )
    generated_text = "".join(vocab.chars[i] for i in generated)

    assert any(word in generated_text for word in ("quick", "fox", "the"))


def test_train_returns_batch_snapshots_as_data():
    vocab = build_vocabulary("abba")
    config = TrainingConfig(hidden_size=3, seq_length=2, seed=0)

    result = train("abba", vocab, config, epochs=2)

    assert isinstance(result.snapshots, tuple)
    assert [snapshot.iteration for snapshot in result.snapshots] == [1, 2]
