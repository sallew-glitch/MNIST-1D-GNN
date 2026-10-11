| Model | Params | Original | Shuffled | Drop | Train time (s) |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 410 | 33.1 ± 0.1 | 33.0 ± 0.1 | 0.1 | 6 |
| MLP | 15,210 | 66.0 ± 1.3 | 64.4 ± 1.1 | 1.6 | 21 |
| GRU (RNN) | 5,134 | 91.7 ± 1.4 | 54.1 ± 1.6 | 37.6 | 287 |
| CNN (repo, 5k params) | 5,210 | 92.6 ± 0.8 | 58.5 ± 0.9 | 34.1 | 44 |
| CNN (wide, 28k params) | 28,298 | 95.3 ± 0.6 | 64.7 ± 1.1 | 30.6 | 105 |
| GNN · random graph | 45,130 | 67.0 ± 0.8 | 65.6 ± 1.6 | 1.4 | 262 |
| GNN · index-order graph | 45,130 | 94.6 ± 0.6 | 66.0 ± 0.9 | 28.6 | 234 |
| GNN · learned graph | 45,130 | 94.3 ± 0.4 | 94.7 ± 0.4 | -0.4 | 242 |
