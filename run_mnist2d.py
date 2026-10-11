"""Bonus: the same pixel-shuffle test on real MNIST (28x28 = 784 nodes).

Usage: python run_mnist2d.py [n_train] [path/to/mnist_idx_folder]
(or set MNIST_DIR; the folder must contain the raw train-/t10k- *.idx*-ubyte files)
"""
import sys, os, json, time
sys.path.insert(0, "src")
import numpy as np
import torch
import torch.nn as nn
from models import MLP, GNN, n_params
from train import train, set_seed

MNIST_DIR = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MNIST_DIR", "data/MNIST")
OUT = "results/runs_mnist2d.jsonl"


def read_idx(name):
    with open(os.path.join(MNIST_DIR, name), "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(shape)


def grid_graph(side=28):
    """8-neighbour pixel grid: what a CNN implicitly assumes."""
    idx = np.arange(side * side).reshape(side, side)
    A = np.zeros((side * side, side * side), np.float32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            src = idx[max(0, -dr):side - max(0, dr), max(0, -dc):side - max(0, dc)]
            dst = idx[max(0, dr):side - max(0, -dr) or None, max(0, dc):side - max(0, -dc) or None]
            A[src.ravel(), dst.ravel()] = 1
    return A


def correlation_graph(x, k=8):
    """Same idea as for MNIST-1D. Always-black border pixels have zero variance,
    so their correlation is undefined; they simply get no edges."""
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.nan_to_num(np.abs(np.corrcoef(x.T)))
    np.fill_diagonal(C, 0)
    nbrs = np.argsort(-C, axis=1)[:, :k]
    A = np.zeros_like(C, dtype=np.float32)
    A[np.repeat(np.arange(len(C)), k), nbrs.ravel()] = 1
    A[C.max(1) == 0] = 0
    return np.maximum(A, A.T)


class CNN2D(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c, 2 * c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(2 * c * 49, 10))

    def forward(self, x):
        return self.net(x.view(-1, 1, 28, 28))


if __name__ == "__main__":
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    x = read_idx("train-images.idx3-ubyte").reshape(-1, 784)[:n_train] / 255.0
    y = read_idx("train-labels.idx1-ubyte")[:n_train].astype(np.int64)
    xt = read_idx("t10k-images.idx3-ubyte").reshape(-1, 784) / 255.0
    yt = read_idx("t10k-labels.idx1-ubyte").astype(np.int64)
    mu, sd = x.mean(), x.std()
    x, xt = ((x - mu) / sd).astype(np.float32), ((xt - mu) / sd).astype(np.float32)
    perm = np.random.default_rng(0).permutation(784)

    steps = 4000
    for ds, (a, b) in {"regular": (x, xt), "shuffled": (x[:, perm], xt[:, perm])}.items():
        models = {
            "MLP": (lambda: MLP(784, 10, hidden=256), 1e-3),
            "CNN": (lambda: CNN2D(), 1e-3),
            "GNN (grid)": (lambda: GNN(grid_graph(), d=32, layers=4), 3e-3),
            "GNN (learned)": (lambda: GNN(correlation_graph(a), d=32, layers=4), 3e-3),
        }
        for name, (make, lr) in models.items():
            set_seed(0)
            m = make()
            r = train(m, a, y, b, yt, steps=steps, lr=lr, bs=100, eval_every=1000)
            rec = dict(dataset=ds, model=name, acc=r["acc"], secs=r["secs"], params=n_params(m), n_train=n_train)
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[mnist2d {ds}] {name:14s} acc={r['acc']:.2f} ({r['secs']:.0f}s)", flush=True)
