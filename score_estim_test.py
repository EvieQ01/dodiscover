"""
Score estimation comparison: KDE, DSM, Gaussian flow, and Diffusion model.

Setup: X1, X2 ~ N(0,1) independent.  Selection: X1+X2 > 0.

True Hessian of log p:
    mean(H_jj) = -1   (constant for Gaussian)
    var(H_jj)  =  0   (constant → zero variance)

The diffusion model (from DiffAN) trains a neural network to predict the
injected noise ε at each timestep t.  The score at timestep t is:

    s(x_t, t) = -ε_θ(x_t, t) / √(1 - ᾱ_t)

and the Hessian of log p_t is the Jacobian of the score:

    H_ij(x_t, t) = ∂s_i/∂x_j = -∂ε_θ_i/∂x_j / √(1 - ᾱ_t)

Key advantage over Stein/KDE: the neural network learns the score globally
and the diffusion smoothing at each t blurs the hard truncation boundary.
Larger t → more smoothing → less boundary distortion, but estimates the
Hessian of a more smoothed distribution (noisier x_t).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DiffAN"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffan.gaussian_diffusion import (
    GaussianDiffusion, UniformSampler, get_named_beta_schedule,
    LossType, ModelMeanType, ModelVarType,
)
from diffan.nn import DiffMLP

# =========================================================================
# Re-use estimators from earlier
# =========================================================================

class KDEHessian:
    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth

    def _bw(self, X):
        if self.bandwidth is not None:
            return self.bandwidth
        n, d = X.shape
        return np.std(X) * (4 / (d + 2) / n) ** (1 / (d + 4))

    def fit(self, X):
        n, d = X.shape
        h = self._bw(X)
        diff = X[:, None, :] - X[None, :, :]
        log_K = -0.5 * (diff**2).sum(2) / h**2
        log_K -= log_K.max(1, keepdims=True)
        K = np.exp(log_K)
        p = K.mean(1)
        dp = -np.einsum("kij,ki->kj", diff, K) / (h**2 * n)
        score = dp / p[:, None]
        outer = np.einsum("kij,kil->kijl", diff, diff)
        d2p = np.einsum("kijl,ki->kjl", outer, K) / (h**4 * n)
        d2p -= K.mean(1)[:, None, None] * np.eye(d) / h**2
        H = d2p / p[:, None, None] - np.einsum("ki,kj->kij", score, score)
        return np.array([H[:, j, j] for j in range(d)]).T, H


class GaussianFlowHessian:
    def fit(self, X):
        n, d = X.shape
        mu = X.mean(0)
        Sigma_inv = np.linalg.inv(np.cov(X.T, ddof=1))
        H_const = -Sigma_inv
        H_full = np.tile(H_const[None], (n, 1, 1))
        return np.array([H_full[:, j, j] for j in range(d)]).T, H_full


# =========================================================================
# Diffusion model score estimator
# =========================================================================

def build_diffusion(n_steps=100):
    betas = get_named_beta_schedule(
        "linear", n_steps, scale=1, beta_start=0.0001, beta_end=0.02
    )
    gd = GaussianDiffusion(
        betas=betas,
        loss_type=LossType.MSE,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_LARGE,
        rescale_timesteps=True,
    )
    return gd


def train_score_model(X_np, gd, epochs=3000, batch_size=256, lr=1e-3, seed=42,
                      early_stop_patience=400, label=""):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    n_nodes = X_np.shape[1]

    # z-score normalise (as in DiffAN.fit)
    mu  = X_np.mean(0, keepdims=True)
    std = X_np.std(0,  keepdims=True)
    X_norm = (X_np - mu) / std

    model = DiffMLP(n_nodes).to(device).float()
    opt   = torch.optim.Adam(model.parameters(), lr)
    sampler = UniformSampler(gd)

    X_t = torch.FloatTensor(X_norm)
    n   = len(X_t)
    val_size   = max(1, int(n * 0.2))
    train_size = n - val_size
    X_train, X_val = X_t[:train_size], X_t[train_size:]

    loader     = DataLoader(X_train, min(train_size, batch_size), drop_last=True)
    val_loader = DataLoader(X_val,   min(val_size,   batch_size))

    best_loss, best_state, best_epoch = float("inf"), None, 0

    pbar = tqdm(range(epochs), desc=f"Train [{label}]", leave=False)
    for epoch in pbar:
        model.train()
        for x_start in loader:
            x_start = x_start.float().to(device)
            t, weights = sampler.sample(x_start.shape[0], device)
            noise = torch.randn_like(x_start)
            x_t   = gd.q_sample(x_start, t, noise=noise)
            pred  = model(x_t, gd._scale_timesteps(t))
            loss  = ((noise - pred)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        if epoch % 50 == 0 and epoch > 200:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for x_start in val_loader:
                    t, _ = sampler.sample(x_start.shape[0], device)
                    noise = torch.randn_like(x_start)
                    x_t   = gd.q_sample(x_start.float(), t, noise=noise)
                    pred  = model(x_t, gd._scale_timesteps(t))
                    val_losses.append(((noise - pred)**2).mean().item())
                val_loss = float(np.mean(val_losses))
            if val_loss < best_loss:
                best_loss  = val_loss
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
            pbar.set_postfix(val=f"{val_loss:.4f}", best_ep=best_epoch)
            if epoch - best_epoch > early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [{label}] stopped ep={epoch}  best_val={best_loss:.4f}  best_ep={best_epoch}")
    return model, mu, std


def diffusion_hessian(model, gd, X_np, mu, std, t_step: int):
    """
    Compute per-sample Hessian of log p_t via Jacobian of the score network.

    Score:   s(x, t) = -ε_θ(x, t) / √(1-ᾱ_t)
    Hessian: H_ij    = ∂s_i/∂x_j
    """
    device = next(model.parameters()).device
    model.eval()

    X_norm = torch.FloatTensor((X_np - mu) / std).to(device)
    n, d   = X_norm.shape
    sqrt_one_minus_ab = float(gd.sqrt_one_minus_alphas_cumprod[t_step])
    t_vec  = (torch.ones(n, dtype=torch.long) * t_step).to(device)

    H_full_list = []
    for i in range(n):
        x_i = X_norm[i:i+1].requires_grad_(True)   # (1, d)
        t_i = t_vec[i:i+1]
        eps = model(x_i, gd._scale_timesteps(t_i))  # (1, d)
        score = -eps / sqrt_one_minus_ab              # (1, d)

        row = []
        for k in range(d):
            g = torch.autograd.grad(score[0, k], x_i, retain_graph=True)[0]
            row.append(g[0].detach().cpu().numpy())
        H_full_list.append(np.stack(row, axis=0))    # (d, d)

    H_full = np.stack(H_full_list, axis=0)           # (n, d, d)
    H_diag = np.stack([H_full[:, j, j] for j in range(d)], axis=1)  # (n, d)
    return H_diag, H_full


# =========================================================================
# Data
# =========================================================================
seed = 42
rng  = np.random.default_rng(seed)
n    = 1000        # keep manageable for neural net training

X_full  = rng.standard_normal((n, 2))
mask    = X_full[:, 0] + X_full[:, 1] > 0
X_trunc = X_full[mask]

print(f"Full: n={len(X_full)}  Truncated: n={len(X_trunc)}")

# =========================================================================
# Train diffusion models
# =========================================================================
gd = build_diffusion(n_steps=100)
print("\nTraining on full data…")
model_full,  mu_f, std_f = train_score_model(X_full,  gd, epochs=3000, label="full")
print("Training on truncated data…")
model_trunc, mu_t, std_t = train_score_model(X_trunc, gd, epochs=3000, label="trunc")

# =========================================================================
# Evaluate Hessian at multiple diffusion timesteps
# =========================================================================
t_steps = [5, 20, 50, 80]
print("\nComputing Jacobians (this may take a minute)…")

results_diff = {}
for t in t_steps:
    Hd_f, Hf_f = diffusion_hessian(model_full,  gd, X_full[:500],  mu_f, std_f, t)
    Hd_t, Hf_t = diffusion_hessian(model_trunc, gd, X_trunc[:500], mu_t, std_t, t)
    results_diff[t] = dict(
        full_mean=Hf_f.mean(0), full_var=Hf_f.var(0),
        trunc_mean=Hf_t.mean(0), trunc_var=Hf_t.var(0),
        Hf_full=Hf_f, Hf_trunc=Hf_t,
    )

# Also compute KDE and Gaussian flow for comparison
kde  = KDEHessian()
gflo = GaussianFlowHessian()

_, Hf_kde_full  = kde.fit(X_full[:500])
_, Hf_kde_trunc = kde.fit(X_trunc[:500])
_, Hf_gf_full   = gflo.fit(X_full[:500])
_, Hf_gf_trunc  = gflo.fit(X_trunc[:500])

# =========================================================================
# Print summary
# =========================================================================
print("\n" + "="*68)
print("Diagonal Hessian variance  (true = 0 for N(0,I))")
print("="*68)
print(f"\n  {'Method':<28} {'var H[0,0] full':>16} {'var H[0,0] trunc':>16} {'ratio':>7}")
print(f"  {'-'*70}")
print(f"  {'True (N(0,I))':<28} {'0':>16} {'0':>16}")
print(f"  {'KDE':<28} {Hf_kde_full[:,0,0].var():>16.4f} "
      f"{Hf_kde_trunc[:,0,0].var():>16.4f} "
      f"{Hf_kde_trunc[:,0,0].var()/Hf_kde_full[:,0,0].var():>7.2f}x")
print(f"  {'Gaussian flow':<28} {Hf_gf_full[:,0,0].var():>16.6f} "
      f"{Hf_gf_trunc[:,0,0].var():>16.6f}")
for t in t_steps:
    r  = results_diff[t]
    vf = r['full_var'][0, 0]
    vt = r['trunc_var'][0, 0]
    ratio = vt / vf if vf > 0 else float('inf')
    print(f"  {'Diffusion t='+str(t):<28} {vf:>16.4f} {vt:>16.4f} {ratio:>7.2f}x")

print(f"\n  {'Method':<28} {'mean H[0,0] full':>16} {'mean H[0,0] trunc':>17}")
print(f"  {'-'*65}")
print(f"  {'True (N(0,I))':<28} {-1:>16.4f} {-1:>17.4f}")
print(f"  {'KDE':<28} {Hf_kde_full[:,0,0].mean():>16.4f} "
      f"{Hf_kde_trunc[:,0,0].mean():>17.4f}")
print(f"  {'Gaussian flow':<28} {Hf_gf_full[:,0,0].mean():>16.4f} "
      f"{Hf_gf_trunc[:,0,0].mean():>17.4f}")
for t in t_steps:
    r = results_diff[t]
    print(f"  {'Diffusion t='+str(t):<28} {r['full_mean'][0,0]:>16.4f} "
          f"{r['trunc_mean'][0,0]:>17.4f}")

# =========================================================================
# Plot
# =========================================================================
fig, axes = plt.subplots(2, len(t_steps)+1, figsize=(15, 7))

# Column 0: KDE reference
for row, (Hf, color, lbl) in enumerate([
        (Hf_kde_full,  "steelblue",  "KDE full"),
        (Hf_kde_trunc, "darkorange", "KDE trunc"),
]):
    ax = axes[row, 0]
    ax.hist(Hf[:, 0, 0], bins=50, color=color, edgecolor="none", alpha=0.8)
    ax.axvline(-1, color="k", ls="--", lw=1.2)
    ax.set_title(f"KDE\n{lbl}", fontsize=8)
    ax.set_xlabel("H[0,0]")
    ax.annotate(f"mean={Hf[:,0,0].mean():+.3f}\nvar={Hf[:,0,0].var():.3f}",
                xy=(0.04, 0.96), xycoords="axes fraction", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

# Columns 1…: diffusion at each t
for col, t in enumerate(t_steps, start=1):
    r = results_diff[t]
    for row, (Hf, color, lbl) in enumerate([
            (r['Hf_full'],  "mediumpurple", f"t={t} full"),
            (r['Hf_trunc'], "crimson",      f"t={t} trunc"),
    ]):
        ax = axes[row, col]
        ax.hist(Hf[:, 0, 0], bins=50, color=color, edgecolor="none", alpha=0.8)
        ax.axvline(-1, color="k", ls="--", lw=1.2)
        ax.set_title(f"Diffusion\n{lbl}", fontsize=8)
        ax.set_xlabel("H[0,0]")
        ax.annotate(f"mean={Hf[:,0,0].mean():+.3f}\nvar={Hf[:,0,0].var():.3f}",
                    xy=(0.04, 0.96), xycoords="axes fraction", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

axes[0, 0].set_ylabel("Full data")
axes[1, 0].set_ylabel("Truncated (X1+X2>0)")
fig.suptitle("Diagonal Hessian distribution: KDE vs Diffusion model at multiple timesteps\n"
             "X1,X2~N(0,1), selection X1+X2>0 — true H[0,0]=-1, true var=0", fontsize=10)
fig.tight_layout()
plt.savefig("score_estim_test.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to score_estim_test.png")
