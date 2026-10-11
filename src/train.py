"""One shared training loop for every model, so the comparison is fair."""
import time
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, x, bs=1000):
    model.eval()
    out = torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)]).argmax(-1)
    model.train()
    return out


def train(model, x, y, x_eval, y_eval, steps=8000, lr=1e-2, bs=100, eval_every=250):
    """Adam + cosine learning-rate decay. Returns final eval accuracy, curve and time.
    We report the *final* accuracy (no picking the best step on the test set)."""
    x, y = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)
    x_eval, y_eval = torch.as_tensor(x_eval, dtype=torch.float32), torch.as_tensor(y_eval, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    loss_fn = nn.CrossEntropyLoss()
    curve = []
    t0 = time.time()
    for step in range(steps):
        if step % (len(x) // bs) == 0:
            order = torch.randperm(len(x))
        idx = order[(step % (len(x) // bs)) * bs:][:bs]
        loss = loss_fn(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % eval_every == 0:
            curve.append((step + 1, (predict(model, x_eval) == y_eval).float().mean().item() * 100))
    secs = time.time() - t0
    preds = predict(model, x_eval).numpy()
    return dict(acc=float((preds == y_eval.numpy()).mean() * 100), curve=curve, secs=secs, preds=preds)
