# Cell 1 (Colab only): install libs (uncomment if running in Colab)
# !pip install torch torchvision opacus pandas matplotlib tqdm

# Cell 2: imports & utils
import time, os, random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Cell 3: hyperparams & data
N = 60000
batch = 256
epochs = 10
sigmas = [0.5, 1.0, 2.0]   # will try these later epochs = 10, sigmas = [0.5, 1.0, 2.0]
clip = 1.0
delta = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=1024, shuffle=False)

# Cell 4: model & eval functions
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

# Cell 5: run experiments (writes results/eps_results.csv)
os.makedirs("results", exist_ok=True)
results = []
for sigma in sigmas:
    seed = 42
    set_seed(seed)
    model = SimpleMLP().to(device)
    opt = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Opacus 1.5.4 API (make_private)
    privacy_engine = PrivacyEngine()
    model, opt, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=opt,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=clip,
    )

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
    eps = privacy_engine.get_epsilon(delta=delta)
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
        "seed": seed
    })
    print(f"Sigma={sigma} -> eps={eps:.3f}, acc={acc:.4f}, time={runtime:.1f}s")

df = pd.DataFrame(results)
df.to_csv("results/eps_results.csv", index=False)
print(df)
