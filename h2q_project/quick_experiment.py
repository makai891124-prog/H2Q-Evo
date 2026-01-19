"""Quick experiment runner for local/fast benchmarking.
This script runs a small-scale training loop adapted from run_experiment.py.
It will use `torch` if available, otherwise falls back to `numpy` for a lightweight simulation.
Outputs a JSON summary to `quick_experiment_result.json`.
"""
import json
import time
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    import numpy as np
    TORCH_AVAILABLE = False

from pathlib import Path

def get_data_batch(batch_size=16):
    if TORCH_AVAILABLE:
        start = torch.randn(batch_size, 1) * 2
        X = torch.cat([start, start + 1, start + 2], dim=1)
        y = start + 3
        return X, y
    else:
        start = np.random.randn(batch_size, 1) * 2
        X = np.concatenate([start, start + 1, start + 2], axis=1)
        y = start + 3
        return X, y

def mse_loss(pred, target):
    if TORCH_AVAILABLE:
        return nn.MSELoss()(pred, target)
    else:
        return float(((pred - target)**2).mean())

def main():
    epochs = 50
    batch_size = 16
    history = {'loss': [], 'autonomy_score': [], 'eta_total': []}

    # Simple param: a scalar weight to predict y = w * mean(X)
    if TORCH_AVAILABLE:
        w = torch.randn(1, requires_grad=True)
        opt = torch.optim.SGD([w], lr=0.01)
    else:
        w = np.random.randn(1)

    cumulative_eta = 0.0
    for ep in range(epochs):
        X, y = get_data_batch(batch_size)
        if TORCH_AVAILABLE:
            pred = torch.mean(X, dim=1, keepdim=True) * w
            loss = mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = float(loss.item())
        else:
            pred = np.mean(X, axis=1, keepdims=True) * w
            loss_val = mse_loss(pred, y)
            # simple gradient step for simulated param
            grad = 2 * ((pred - y) * np.mean(X, axis=1, keepdims=True)).mean()
            w = w - 0.01 * grad

        # simulated metrics
        eta_step = 0.01 * (0.5 + 0.5 * (1.0 / (1.0 + ep)))
        cumulative_eta += eta_step
        autonomy_score = max(0.0, min(1.0, 0.1 + 0.9 * (1.0 / (1.0 + loss_val))))

        history['loss'].append(float(loss_val))
        history['eta_total'].append(float(cumulative_eta))
        history['autonomy_score'].append(float(autonomy_score))

        if ep % 10 == 0:
            print(f"epoch {ep}: loss={loss_val:.4f}, eta={cumulative_eta:.4f}, autonomy={autonomy_score:.3f}")

    result = {
        'torch_available': TORCH_AVAILABLE,
        'final_w': (float(w.item()) if TORCH_AVAILABLE else float(w[0])),
        'history': history,
        'epochs': epochs,
        'batch_size': batch_size,
        'runtime_seconds': 0
    }

    outp = Path('quick_experiment_result.json')
    outp.write_text(json.dumps(result, indent=2))
    print('\nSaved quick_experiment_result.json')

if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"Done in {time.time()-t0:.2f}s")
