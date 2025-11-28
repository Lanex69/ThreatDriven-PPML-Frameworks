# appendix_A_final_run.py
"""
Polished DP-SGD MNIST worked example (for paper Appendix A).
Run: python appendix_A_final_run.py
Saves: results/eps_results_final.csv
"""
import time, os, random
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Polished hyperparams
batch = 128
epochs = 15
sigmas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]  # you can adjust
clip = 1.0
delta = 1e-5
seeds = [42, 43, 44]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
N = len(train_ds)

num_workers = 2 if torch.cuda.is_available() else 0
pin_memory = True if torch.cuda.is_available() else False
train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
test_loader  = DataLoader(test_ds, batch_size=1024, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.net(x)

def test_acc(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds==y).sum().item()
            total += y.size(0)
    return correct/total

results = []

# non-private baseline
print("RUNNING NON-PRIVATE BASELINE")
for seed in seeds:
    set_seed(seed)
    model = SimpleMLP().to(device)
    opt = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for _ in range(epochs):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
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
        "batch": batch,
        "steps": int(np.ceil(N/batch)*epochs),
        "delta": delta,
        "epsilon": np.nan,
        "test_acc": float(acc),
        "runtime_s": float(runtime),
        "seed": seed,
        "opacus_version": getattr(opacus, "__version__", None),
        "torch_version": torch.__version__
    })
    print(f"[baseline] seed={seed} acc={acc:.4f} time={runtime:.1f}s")

# DP runs
if PrivacyEngine is None:
    print("Opacus not available — skipping DP runs. Install opacus to run DP experiments.")
else:
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
                for x,y in private_train_loader:
                    x,y = x.to(device), y.to(device)
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
                "batch": batch,
                "steps": int(np.ceil(N/batch)*epochs),
                "delta": delta,
                "epsilon": float(eps),
                "test_acc": float(acc),
                "runtime_s": float(runtime),
                "seed": seed,
                "opacus_version": getattr(opacus, "__version__", None),
                "torch_version": torch.__version__
            })
            print(f"Sigma={sigma} seed={seed} -> eps={eps:.3f}, acc={acc:.4f}, time={runtime:.1f}s")

df = pd.DataFrame(results)
df.to_csv("results/eps_results_final.csv", index=False)
print("Wrote results/eps_results_final.csv")
print(df)
