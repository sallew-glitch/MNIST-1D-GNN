"""Main experiment: every model x {regular, shuffled} x 5 seeds, evaluated on the test set.

Usage: python run_experiments.py [regular|shuffled]   (default: both)
Results are appended to results/runs_<dataset>.jsonl as they finish (safe to interrupt + resume).
"""
import sys, os, json
sys.path.insert(0, "src")
from data import load
from graphs import chain_graph, correlation_graph, random_graph
from models import Linear, MLP, CNN, GRU, GNN, n_params
from train import train, set_seed

SEEDS = [0, 1, 2, 3, 4]
STEPS = 8000

D = load()
DATASETS = {"regular": (D["x"], D["x_test"]), "shuffled": (D["xs"], D["xs_test"])}


def models_for(x_train):
    """name -> (constructor, learning rate). Graphs are built from training data only."""
    A_chain = chain_graph()              # "index order == spatial order"
    A_corr = correlation_graph(x_train)  # learned from data correlations
    A_rand = random_graph()              # control
    return {
        "Logistic":        (lambda: Linear(), 1e-2),
        "MLP":             (lambda: MLP(), 1e-2),
        "CNN":             (lambda: CNN(), 1e-2),
        "CNN (wide)":      (lambda: CNN(channels=64), 1e-2),
        "GRU":             (lambda: GRU(), 1e-2),
        "GNN (chain)":     (lambda: GNN(A_chain, d=64, layers=4), 3e-3),
        "GNN (random)":    (lambda: GNN(A_rand, d=64, layers=4), 3e-3),
        "GNN (learned)":   (lambda: GNN(A_corr, d=64, layers=4), 3e-3),
    }


def done_keys(out):
    if not os.path.exists(out):
        return set()
    return {(r["model"], r["seed"]) for r in map(json.loads, open(out))}


if __name__ == "__main__":
    which = sys.argv[1:] or list(DATASETS)
    for ds in which:
        out = f"results/runs_{ds}.jsonl"
        done = done_keys(out)
        x_tr, x_te = DATASETS[ds]
        for name, (make, lr) in models_for(x_tr).items():
            for seed in SEEDS:
                if (name, seed) in done:
                    continue
                set_seed(seed)
                m = make()
                r = train(m, x_tr, D["y"], x_te, D["y_test"], steps=STEPS, lr=lr)
                rec = dict(dataset=ds, model=name, seed=seed, acc=r["acc"], secs=r["secs"],
                           params=n_params(m), curve=r["curve"], preds=r["preds"].tolist())
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{ds}] {name:14s} seed={seed} acc={r['acc']:.1f} ({r['secs']:.0f}s)", flush=True)
