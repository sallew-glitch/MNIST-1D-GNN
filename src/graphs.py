"""Ways to turn 40 signal positions into a graph (40 nodes, adjacency matrix)."""
import numpy as np


def chain_graph(n=40, radius=2):
    """Assume index order == spatial order: connect i to i±1 ... i±radius.
    This is exactly the assumption a CNN makes."""
    i = np.arange(n)
    d = np.abs(i[:, None] - i[None, :])
    return ((d > 0) & (d <= radius)).astype(np.float32)


def correlation_graph(x_train, k=4):
    """Learn the graph from data: connect each position to the k positions whose
    values are most correlated with it across the training set.
    Uses only training data and never looks at index order, so it gives the
    same graph (up to relabelling) whether or not the columns are shuffled."""
    C = np.abs(np.corrcoef(x_train.T))
    np.fill_diagonal(C, 0)
    nbrs = np.argsort(-C, axis=1)[:, :k]
    A = np.zeros_like(C, dtype=np.float32)
    A[np.repeat(np.arange(len(C)), k), nbrs.ravel()] = 1
    return np.maximum(A, A.T)  # make it undirected


def random_graph(n=40, k=4, seed=0):
    """Control: same number of edges per node, but random. Tests whether the
    *structure* matters or just having message passing at all."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        A[i, rng.choice(np.delete(np.arange(n), i), k, replace=False)] = 1
    return np.maximum(A, A.T)
