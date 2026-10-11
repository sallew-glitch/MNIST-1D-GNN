"""Build figures + results table from results/runs_*.jsonl."""
import sys, json, os
sys.path.insert(0, "src")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from data import load
from graphs import correlation_graph, chain_graph

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False,
    "axes.spines.right": False})

ORDER = ["Logistic", "MLP", "GRU", "CNN", "CNN (wide)", "GNN (random)", "GNN (chain)", "GNN (learned)"]
LABEL = {"Logistic": "Logistic regression", "MLP": "MLP", "GRU": "GRU (RNN)", "CNN": "CNN (repo, 5k params)",
         "CNN (wide)": "CNN (wide, 28k params)", "GNN (random)": "GNN · random graph",
         "GNN (chain)": "GNN · index-order graph", "GNN (learned)": "GNN · learned graph"}


def load_runs():
    runs = []
    for ds in ["regular", "shuffled"]:
        p = f"results/runs_{ds}.jsonl"
        if os.path.exists(p):
            runs += [json.loads(l) for l in open(p)]
    return runs


def summarize(runs):
    S = {}
    for r in runs:
        S.setdefault((r["model"], r["dataset"]), []).append(r)
    out = {}
    for (m, ds), rs in S.items():
        a = np.array([r["acc"] for r in rs])
        out[(m, ds)] = dict(mean=a.mean(), std=a.std(), n=len(a), params=rs[0]["params"],
                            secs=np.mean([r["secs"] for r in rs]))
    return out


def hero(S, path="figures/fig1_accuracy_dumbbell.png"):
    models = [m for m in ORDER if (m, "regular") in S and (m, "shuffled") in S]
    fig, ax = plt.subplots(figsize=(10, 6.6), dpi=200)
    ys = np.arange(len(models))[::-1]
    for y, m in zip(ys, models):
        r, s = S[(m, "regular")], S[(m, "shuffled")]
        is_gnn = m == "GNN (learned)"
        if is_gnn:
            ax.axhspan(y - 0.42, y + 0.42, color="#eaf2fc", zorder=0, lw=0)
        ax.plot([s["mean"], r["mean"]], [y, y], color="#c9c8c2", lw=2, solid_capstyle="round", zorder=1)
        for st, c in [(r, BLUE), (s, ORANGE)]:
            ax.errorbar(st["mean"], y, xerr=st["std"], fmt="none", ecolor=c, elinewidth=1.2, capsize=0, zorder=2)
            ax.scatter(st["mean"], y, s=90, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        drop = r["mean"] - s["mean"]
        txt = (f"−{drop:.0f} pt" + ("s" if round(drop) > 1 else "")) if drop >= 1 else "no drop"
        if m in ("GNN (learned)", "CNN (wide)"):
            lo = min(r["mean"], s["mean"])
            ax.text(lo - 1.8, y, f"{r['mean']:.1f}% → {s['mean']:.1f}%", va="center", ha="right", fontsize=10,
                    color=INK, fontweight="bold" if is_gnn else "normal")
        ax.text(102, y, txt, va="center", ha="left", fontsize=10.5,
                color=INK if is_gnn or drop >= 10 else INK2, fontweight="bold" if is_gnn else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[m] for m in models], fontsize=11)
    for t, m in zip(ax.get_yticklabels(), models):
        if m == "GNN (learned)":
            t.set_fontweight("bold"); t.set_color(INK)
    ax.set_xlim(20, 100)
    ax.set_xticks(range(20, 101, 10))
    ax.set_xticklabels([f"{v}%" for v in range(20, 101, 10)])
    ax.xaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Test accuracy (mean ± std over 5 seeds)")
    ax.text(102, ys[0] + 0.45, "drop", ha="left", va="bottom", fontsize=9.5, color=MUTED)
    # legend
    ax.scatter([], [], s=90, color=BLUE, edgecolor=SURFACE, lw=2, label="Original MNIST-1D")
    ax.scatter([], [], s=90, color=ORANGE, edgecolor=SURFACE, lw=2, label="Shuffled MNIST-1D (positions scrambled)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=10.5,
              handletextpad=0.3, columnspacing=1.5)
    fig.suptitle("Scramble the input order: CNNs collapse, a GNN with a learned graph doesn't",
                 x=0.02, ha="left", fontsize=15, fontweight="bold", y=0.985)
    fig.text(0.02, 0.025, "MNIST-1D (Greydanus & Steinbach). Same 4,000 train / 1,000 test samples; 'shuffled' applies one fixed "
             "permutation to the 40 positions.\nLearned graph = connect each position to its 4 most-correlated positions "
             "in the training set (no position info used).", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.24, right=0.9, top=0.86, bottom=0.17)
    fig.savefig(path)
    plt.close(fig)


def ring_order(A):
    L = np.diag(A.sum(1)) - A
    _, v = np.linalg.eigh(L)
    return np.argsort(np.arctan2(v[:, 2], v[:, 1]))


def unscramble(D, path="figures/fig2_learned_graph.png"):
    A_shuf = correlation_graph(D["xs"])
    order = ring_order(A_shuf)
    perm = D["perm"]  # shuffled column j == true position perm[j]
    true_of_order = perm[order]
    # align direction/rotation so the recovered sequence is shown in its natural orientation
    if np.corrcoef(np.arange(40), np.unwrap(true_of_order * 2 * np.pi / 40))[0, 1] < 0:
        order = order[::-1]
    start = int(np.argmin(perm[order]))
    order = np.roll(order, -start)

    i = int(np.where(D["y"] == 3)[0][0])
    fig = plt.figure(figsize=(12, 7.2), dpi=200)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.55, wspace=0.28)
    sig = [(D["x"][i], "1. Original signal (digit 3)", BLUE),
           (D["xs"][i], "2. Shuffled — what the model sees", ORANGE),
           (D["xs"][i][order], "3. Re-ordered using only the learned graph", AQUA)]
    for k, (s, title, c) in enumerate(sig):
        ax = fig.add_subplot(gs[0, k])
        ax.plot(s, color=c, lw=2, solid_joinstyle="round")
        ax.set_title(title, fontsize=11, loc="left", color=INK)
        ax.set_xticks([0, 20, 39]); ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("position", fontsize=9)
    # adjacency matrices
    mats = [(chain_graph(), "Index-order graph\n(what a CNN assumes)"),
            (A_shuf, "Learned graph on shuffled data\n(rows in scrambled order)"),
            (A_shuf[np.ix_(order, order)], "Same learned graph, rows re-ordered\n→ a band + corners: a ring!")]
    for k, (M, title) in enumerate(mats):
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1.6, interpolation="nearest")
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(GRID)
    fig.suptitle("The GNN's learned graph un-scrambles the signal — and reveals it is circular",
                 x=0.02, ha="left", fontsize=15, fontweight="bold", y=0.99)
    true_nb = sum(A_shuf[a, b] for a in range(40) for b in range(a)
                  if min(abs(perm[a] - perm[b]), 40 - abs(perm[a] - perm[b])) <= 2)
    fig.text(0.02, 0.015, f"{int(true_nb)} of {int(A_shuf.sum() / 2)} learned edges connect positions that are truly "
             "within 2 steps of each other (counting wrap-around). Re-ordering = angle of the 2nd/3rd Laplacian eigenvectors.\nThe corners appear "
             "because MNIST-1D's random translation wraps around (np.roll-style), so position 39 really neighbours position 0.",
             fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.88, bottom=0.08)
    fig.savefig(path)
    plt.close(fig)
    return int(true_nb), int(A_shuf.sum() / 2), perm[order]


def table(S, path="results/results_table.md"):
    lines = ["| Model | Params | Original | Shuffled | Drop | Train time (s) |", "|---|---:|---:|---:|---:|---:|"]
    for m in ORDER:
        if (m, "regular") not in S or (m, "shuffled") not in S:
            continue
        r, s = S[(m, "regular")], S[(m, "shuffled")]
        lines.append(f"| {LABEL[m]} | {r['params']:,} | {r['mean']:.1f} ± {r['std']:.1f} | "
                     f"{s['mean']:.1f} ± {s['std']:.1f} | {r['mean'] - s['mean']:.1f} | {r['secs']:.0f} |")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    S = summarize(load_runs())
    table(S)
    hero(S)
    print(unscramble(load()))
