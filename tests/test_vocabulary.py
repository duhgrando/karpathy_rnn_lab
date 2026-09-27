"""§ 'Character-Level Language Models' -- 1-of-k encoding of a small vocabulary."""
import numpy as np

from domain.vocabulary import build_vocabulary, decode, encode, one_hot


def test_vocabulary_is_derived_from_corpus_characters():
    vocab = build_vocabulary("hello")
    assert set(vocab.chars) == {"h", "e", "l", "o"}


def test_encode_decode_roundtrip():
    vocab = build_vocabulary("hello")
    text = "hello"
    assert decode(vocab, encode(vocab, text)) == text


def test_one_hot_matches_the_articles_1_of_k_description():
    vocab = build_vocabulary("helo")
    index = vocab.chars.index("l")
    vector = one_hot(vocab, index)
    assert vector.shape == (vocab.size, 1)
    assert np.count_nonzero(vector) == 1
    assert vector[index, 0] == 1.0
