# train_approx.py  (à lancer avec l'environnement tp)
import numpy as np
import torch, torch.nn as nn
from pathlib import Path

R = 5.0
N = 100
STEPS = 5000
WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256]
OUT = Path("results")
OUT.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Grille (même convention que np.meshgrid)
g = torch.linspace(-R, R, N)
Xg, Yg = torch.meshgrid(g, g, indexing="xy")
pts = torch.stack([Xg.ravel(), Yg.ravel()], dim=1).to(device)
target = torch.clamp(pts.max(dim=1).values, min=0).unsqueeze(1)

X, Y = Xg.numpy(), Yg.numpy()
Z_exact = np.maximum(np.maximum(X, 0), Y)

def train(n):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, n), nn.ReLU(), nn.Linear(n, 1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = ((model(pts) - target) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return model(pts).cpu().numpy().reshape(N, N)

# Grille et surface exacte (pour que le livre n'ait pas à les recalculer)
np.save(OUT / "grid_X.npy", X)
np.save(OUT / "grid_Y.npy", Y)
np.save(OUT / "Z_exact.npy", Z_exact)

# Un fichier par largeur, plus un fichier pour les erreurs max
max_err = []
for n in WIDTHS:
    Z_hat = train(n)
    np.save(OUT / f"Z_hat_n{n}.npy", Z_hat)
    err = np.abs(Z_hat - Z_exact).max()
    max_err.append(err)
    print(f"n={n:4d}  erreur max = {err:.4f}")

np.save(OUT / "widths.npy", np.array(WIDTHS))
np.save(OUT / "max_err.npy", np.array(max_err))