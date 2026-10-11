"""Control experiment: un-scramble the shuffled data using only the learned graph, then train a plain CNN.

Usage: python run_reordered_cnn.py   -> results/cnn_reordered.json
"""
import sys, json
sys.path.insert(0, "src")
import numpy as np
from data import load
from graphs import correlation_graph
from models import CNN
from train import train, set_seed
from make_figures import ring_order

D = load()
order = ring_order(correlation_graph(D["xs"]))  # recovered from shuffled training data only
accs = []
for seed in range(3):
    set_seed(seed)
    r = train(CNN(channels=64), D["xs"][:, order], D["y"], D["xs_test"][:, order], D["y_test"], steps=8000, lr=1e-2)
    accs.append(r["acc"])
    print(f"seed={seed} acc={r['acc']:.1f}", flush=True)
json.dump(dict(model="CNN (wide) on re-ordered shuffled data", accs=accs), open("results/cnn_reordered.json", "w"))
print(f"mean {np.mean(accs):.1f} ± {np.std(accs):.1f}")
