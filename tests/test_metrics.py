import numpy as np
from sklearn.metrics import f1_score

from net.metrics.metrics import compute_metrics


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = np.arange(200) % 10
    probs = rng.dirichlet(np.ones(10), 200)
    m = compute_metrics(y, probs)
    assert np.isclose(m["macro_f1"], f1_score(y, probs.argmax(1), average="macro"))
    assert np.allclose([m[f"f1_{k}"] for k in range(10)], f1_score(y, probs.argmax(1), average=None))
