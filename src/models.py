"""Baselines copied from the MNIST-1D repo (notebooks/models.py) + a GNN."""
import torch
import torch.nn as nn


class Linear(nn.Module):  # logistic regression
    def __init__(self, n_in=40, n_out=10):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, n_in=40, n_out=10, hidden=100):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(n_in, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, n_out)

    def forward(self, x):
        h = self.l1(x).relu()
        h = h + self.l2(h).relu()
        return self.l3(h)


class CNN(nn.Module):
    def __init__(self, n_out=10, channels=25):
        super().__init__()
        self.conv1 = nn.Conv1d(1, channels, 5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.linear = nn.Linear(channels * 5, n_out)

    def forward(self, x):
        h = x.view(-1, 1, x.shape[-1])
        h = self.conv3(self.conv2(self.conv1(h).relu()).relu()).relu()
        return self.linear(h.flatten(1))


class GRU(nn.Module):
    def __init__(self, n_out=10, hidden=6, steps=40):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * hidden * steps, n_out)

    def forward(self, x):
        out, _ = self.gru(x.unsqueeze(-1))
        return self.linear(out.flatten(1))


class GraphConv(nn.Module):
    """One round of message passing (GraphSAGE-style):
        h_i  <-  W_self · h_i  +  W_nbr · mean_{j in neighbours(i)} h_j
    Every node looks at its neighbours and updates itself."""

    def __init__(self, d):
        super().__init__()
        self.self_lin = nn.Linear(d, d)
        self.nbr_lin = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, A_norm):  # h: (batch, nodes, d), A_norm: (nodes, nodes)
        return h + self.norm(self.self_lin(h) + self.nbr_lin(A_norm @ h)).relu()


class GNN(nn.Module):
    """Each of the 40 positions is a node; its feature is the signal value.
    Node identity (which position it is) is a learned embedding, so the model
    never relies on index order -- only on the graph it is given."""

    def __init__(self, A, n_out=10, d=32, layers=4):
        super().__init__()
        A = torch.as_tensor(A, dtype=torch.float32)
        self.register_buffer("A_norm", A / A.sum(1, keepdim=True).clamp(min=1))
        n = A.shape[0]
        self.embed_val = nn.Linear(1, d)
        self.embed_node = nn.Parameter(torch.randn(n, d) * 0.1)
        self.convs = nn.ModuleList([GraphConv(d) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, n_out))

    def forward(self, x):
        h = self.embed_val(x.unsqueeze(-1)) + self.embed_node  # (B, N, d)
        for conv in self.convs:
            h = conv(h, self.A_norm)
        g = torch.cat([h.mean(1), h.amax(1)], -1)  # graph-level readout
        return self.head(g)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
