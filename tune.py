"""Pick hyperparameters on a validation split (last 500 training samples). Test set untouched."""
import sys, json
sys.path.insert(0, "src")
from data import load
from graphs import chain_graph, correlation_graph
from models import Linear, MLP, CNN, GRU, GNN, n_params
from train import train, set_seed

D = load()
xtr, ytr, xva, yva = D["x"][:3500], D["y"][:3500], D["x"][3500:], D["y"][3500:]
res = {}
def run(name, make, lr):
    set_seed(0); m = make()
    r = train(m, xtr, ytr, xva, yva, steps=6000, lr=lr)
    res[f"{name} lr={lr}"] = r["acc"]
    print(f"{name:28s} lr={lr:<6} params={n_params(m):6d} val={r['acc']:.1f}  ({r['secs']:.0f}s)", flush=True)

for lr in [1e-2, 3e-3]:
    run("linear", Linear, lr); run("mlp", MLP, lr); run("cnn", CNN, lr)
Ac, Ak = chain_graph(), correlation_graph(xtr)
for lr in [3e-3, 1e-3]:
    for d, L in [(32, 4), (64, 4), (64, 6)]:
        run(f"gnn-chain d={d} L={L}", lambda: GNN(Ac, d=d, layers=L), lr)
        run(f"gnn-corr d={d} L={L}", lambda: GNN(Ak, d=d, layers=L), lr)
run("gru", GRU, 1e-2)
json.dump(res, open("results/tuning.json", "w"), indent=1)
