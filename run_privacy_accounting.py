# run_privacy_accounting.py
"""
Polished DP-SGD MNIST worked example (for paper Appendix A).
Run (examples):
    python run_privacy_accounting.py
    python run_privacy_accounting.py --out results/eps_results_demo.csv --quick
    python run_privacy_accounting.py --epochs 10 --batch 64 --sigmas 0.5 1.0 2.0

Saves: results/eps_results.csv (by default)
"""
import time
import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

try:
    from opacus import PrivacyEngine
    import opacus
except Exception:
    PrivacyEngine = None
    opacus = None
    print("Warning: opacus not available. Install opacus to run DP experiments.")


def parse_args():
    p = argparse.ArgumentParser(description="DP-SGD MNIST worked example (Appendix A)")
    p.add_argument("--out", type=str, default="results/eps_results.csv",
                   help="Path to output CSV (default: results/eps_results.csv)")
    p.add_argument("--quick", action="store_true",
                   help="Run a short demo run (1 epoch, 1 seed, single sigma) for notebooks/CI")
    p.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    p.add_argument("--batch", type=int, default=128, help="Training batch size (default: 128)")
    p.add_argument("--sigmas", type=float, nargs="*", default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
                   help="List of noise multipliers (sigma) for DP runs")
    p.add_argument("--clip", type=float, default=1.0, help="Max grad norm clipping (default: 1.0)")
    p.add_argument("--delta", type=float, default=1e-5, help="Delta for DP accounting (default: 1e-5)")
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44],
                   help="Random seeds to run (default: 42 43 44)")
    p.add_argument("--device", type=str, default=None,
                   help="Device to use, e.g. 'cpu' or 'cuda'. If not set auto-detects.")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


def test_acc(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def run_non_private(seeds, epochs, train_loader, test_loader, device, clip, delta):
    results = []
    N = len(train_loader.dataset)
    for seed in seeds:
        set_seed(seed)
        model = SimpleMLP().to(device)
        opt = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        start = time.time()
        for _ in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                opt.step()
        runtime = time.time() - start
        acc = test_acc(model, test_loader, device)

        results.append({
            "sigma": None,
            "clip": clip,
            "epochs": epochs,
            "batch": train_loader.batch_size,
            "steps": int(np.ceil(N / train_loader.batch_size) * epochs),
            "delta": delta,
            "epsilon": np.nan,
            "test_acc": float(acc),
            "runtime_s": float(runtime),
            "seed": seed,
            "opacus_version": getattr(opacus, "__version__", None),
            "torch_version": torch.__version__
        })
        print(f"[baseline] seed={seed} acc={acc:.4f} time={runtime:.1f}s")
    return results


def run_dp_runs(sigmas, seeds, epochs, train_loader, test_loader, device, clip, delta):
    results = []
    N = len(train_loader.dataset)
    if PrivacyEngine is None:
        print("Opacus not available — skipping DP runs. Install opacus to run DP experiments.")
        return results

    for sigma in sigmas:
        print(f"\nRUNNING DP RUNS sigma={sigma}")
        for seed in seeds:
            set_seed(seed)
            model = SimpleMLP().to(device)
            opt = optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            # For final audited epsilon set secure_mode=True (requires torchcsprng)
            # privacy_engine = PrivacyEngine(secure_mode=True)
            privacy_engine = PrivacyEngine()
            model, opt, private_train_loader = privacy_engine.make_private(
                module=model,
                optimizer=opt,
                data_loader=train_loader,
                noise_multiplier=sigma,
                max_grad_norm=clip,
            )

            start = time.time()
            for _ in range(epochs):
                model.train()
                for x, y in private_train_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = criterion(model(x), y)
                    loss.backward()
                    opt.step()
            runtime = time.time() - start

            try:
                eps = privacy_engine.get_epsilon(delta=delta)
            except Exception:
                try:
                    eps = privacy_engine.accountant.get_epsilon(delta)
                except Exception:
                    eps = float("nan")

            acc = test_acc(model, test_loader, device)

            results.append({
                "sigma": sigma,
                "clip": clip,
                "epochs": epochs,
                "batch": train_loader.batch_size,
                "steps": int(np.ceil(N / train_loader.batch_size) * epochs),
                "delta": delta,
                "epsilon": float(eps),
                "test_acc": float(acc),
                "runtime_s": float(runtime),
                "seed": seed,
                "opacus_version": getattr(opacus, "__version__", None),
                "torch_version": torch.__version__
            })
            print(f"Sigma={sigma} seed={seed} -> eps={eps:.3f}, acc={acc:.4f}, time={runtime:.1f}s")
    return results


def main():
    args = parse_args()

    # Quick mode overrides
    if args.quick:
        print("Quick demo mode enabled: using 1 epoch, 1 seed, sigma=1.0")
        epochs = 1
        seeds = [42]
        sigmas = [1.0]
    else:
        epochs = args.epochs
        seeds = args.seeds
        sigmas = args.sigmas

    batch = args.batch
    clip = args.clip
    delta = args.delta
    out_csv = args.out

    # Device selection
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare directories
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Data loaders (respecting batch size)
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    N = len(train_ds)

    num_workers = 2 if torch.cuda.is_available() else 0
    pin_memory = True if torch.cuda.is_available() else False
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    # Run baseline and DP runs
    all_results = []
    print("RUNNING NON-PRIVATE BASELINE")
    all_results.extend(run_non_private(seeds, epochs, train_loader, test_loader, device, clip, delta))
    all_results.extend(run_dp_runs(sigmas, seeds, epochs, train_loader, test_loader, device, clip, delta))

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(df)


if __name__ == "__main__":
    main()
