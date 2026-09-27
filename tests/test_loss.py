"""§ 'A more technical explanation' -- the Softmax / cross-entropy loss."""
import numpy as np

from domain.rnn_model import cross_entropy_loss, softmax


def test_uniform_prediction_gives_the_maximum_entropy_loss():
    """With no learning at all, softmax(0) is uniform, so the loss of a
    single correct-character prediction should equal -log(1 / vocab_size)."""
    vocab_size = 4
    uniform = softmax(np.zeros((vocab_size, 1)))

    loss = cross_entropy_loss([uniform], [2])

    assert np.isclose(loss, -np.log(1.0 / vocab_size), atol=1e-6)


def test_confident_correct_prediction_gives_low_loss():
    p = np.array([[0.01], [0.01], [0.97], [0.01]])
    assert cross_entropy_loss([p], [2]) < 0.05


def test_confident_wrong_prediction_gives_high_loss():
    p = np.array([[0.97], [0.01], [0.01], [0.01]])
    assert cross_entropy_loss([p], [2]) > 3.0


def test_loss_accumulates_additively_over_a_sequence():
    p = np.array([[0.5], [0.5]])
    single_step_loss = cross_entropy_loss([p], [0])
    three_step_loss = cross_entropy_loss([p, p, p], [0, 0, 0])
    assert np.isclose(three_step_loss, 3 * single_step_loss)
