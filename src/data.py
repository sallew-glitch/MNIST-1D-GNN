"""Load MNIST-1D (regular + spatially shuffled), using the official generator."""
import os
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args

CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "mnist1d.npz")


def load():
    """Returns dict with x, y, x_test, y_test, xs, xs_test (shuffled), perm.

    The shuffled dataset is generated exactly the way the repo does it
    (shuffle_seq=True): one fixed random permutation of the 40 positions,
    applied to every sample. We also recover that permutation so we can
    later check whether a model "rediscovered" the true ordering.
    """
    if os.path.exists(CACHE):
        return dict(np.load(CACHE))
    a = get_dataset_args()
    d = make_dataset(a)
    a.shuffle_seq = True
    s = make_dataset(a)
    assert (d["y"] == s["y"]).all()
    # column j of the shuffled data == column perm[j] of the regular data
    perm = np.array([np.argmin(((d["x"] - s["x"][:, [j]]) ** 2).sum(0)) for j in range(40)])
    assert np.allclose(d["x"][:, perm], s["x"])
    out = dict(x=d["x"], y=d["y"], x_test=d["x_test"], y_test=d["y_test"],
               xs=s["x"], xs_test=s["x_test"], perm=perm)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, **out)
    return out
