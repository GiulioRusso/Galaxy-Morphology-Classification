import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from net.metrics.metrics import compute_metrics as metrics
from net.metrics.metrics import ece


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = np.arange(200) % 10
    probs = rng.dirichlet(np.ones(10), 200)
    pred = probs.argmax(1)
    m = metrics(y, probs)
    assert np.isclose(m["macro_f1"], f1_score(y, pred, average="macro"))
    assert np.isclose(m["balanced_acc"], balanced_accuracy_score(y, pred))
    assert np.isclose(m["acc"], accuracy_score(y, pred))


def test_ece_toy():
    y = np.array([0, 0, 1, 1])
    probs = np.array([[1.0, 0], [0, 1.0], [0, 1.0], [0, 1.0]])  # conf 1, acc 3/4
    assert np.isclose(ece(y, probs), 0.25)


def test_superclass_sums_probabilities():
    # class 0 and 1 share superclass 0: argmax class is 2, but superclass 0 has more mass
    probs = np.array([[0.3, 0.3, 0.4, 0, 0, 0, 0, 0, 0, 0]])
    assert metrics(np.array([0]), probs)["super4_acc"] == 1.0
