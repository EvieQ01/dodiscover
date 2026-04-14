"""
Score estimation comparison: Stein, KDE, DSM, Gaussian flow, Diffusion model.

Setup: X1, X2 ~ N(0,1) independent.  Selection: X1+X2 > 0.

True Hessian of log p:
    mean(H_jj) = -1   (constant for Gaussian)
    var(H_jj)  =  0   (constant → zero variance)

Five methods are compared on diagonal Hessian variance before/after truncation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DiffAN"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from functorch import vmap, jacrev
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffan.gaussian_diffusion import (
    GaussianDiffusion, UniformSampler, get_named_beta_schedule,
    LossType, ModelMeanType, ModelVarType,
)
from diffan.nn import DiffMLP
from dodiscover.toporder._base import SteinMixin

# =========================================================================
# Estimators
# =========================================================================

class SteinHessian:
    def __init__(self, eta_G=0.001, eta_H=0.001):
        self.eta_G = eta_G
        self.eta_H = eta_H
        self._stein = SteinMixin()

    def fit(self, X):
        H = self._stein.hessian(X, self.eta_G, self.eta_H)   # (n, d, d)
        return np.array([H[:, j, j] for j in range(X.shape[1])]).T, H


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


class DSMHessian:
    """Nadaraya-Watson denoising score matching Hessian."""
    def __init__(self, sigma=0.3, seed=0):
        self.sigma = sigma
        self.seed = seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        s = self.sigma
        X_eval = X + rng.standard_normal(X.shape) * s
        diff = X_eval[:, None, :] - X[None, :, :]
        log_K = -0.5 * (diff**2).sum(2) / s**2
        log_K -= log_K.max(1, keepdims=True)
        K = np.exp(log_K)
        W = K / K.sum(1, keepdims=True)
        NW = np.einsum("ki,ij->kj", W, X)
        EW_xx = np.einsum("ki,ij,il->kjl", W, X, X)
        CovW = EW_xx - np.einsum("kj,kl->kjl", NW, NW)
        H = CovW / s**4 - np.eye(d) / s**2
        return np.array([H[:, j, j] for j in range(d)]).T, H


class GaussianFlowHessian:
    def fit(self, X):
        n, d = X.shape
        Sigma_inv = np.linalg.inv(np.cov(X.T, ddof=1))
        H_full = np.tile(-Sigma_inv[None], (n, 1, 1))
        return np.array([H_full[:, j, j] for j in range(d)]).T, H_full


# =========================================================================
# Diffusion model
# =========================================================================

def build_diffusion(n_steps=100):
    betas = get_named_beta_schedule(
        "linear", n_steps, scale=1, beta_start=0.0001, beta_end=0.02
    )
    return GaussianDiffusion(
        betas=betas, loss_type=LossType.MSE,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_LARGE,
        rescale_timesteps=True,
    )


def train_score_model(X_np, gd, epochs=3000, batch_size=256, lr=1e-3,
                      seed=42, patience=400, label=""):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    mu  = X_np.mean(0, keepdims=True)
    std = X_np.std(0,  keepdims=True)
    X_norm = (X_np - mu) / std

    model   = DiffMLP(X_np.shape[1]).to(device).float()
    opt     = torch.optim.Adam(model.parameters(), lr)
    sampler = UniformSampler(gd)

    X_t = torch.FloatTensor(X_norm)
    val_size = max(1, int(len(X_t) * 0.2))
    X_train, X_val = X_t[:-val_size], X_t[-val_size:]
    loader     = DataLoader(X_train, min(len(X_train), batch_size), drop_last=True)
    val_loader = DataLoader(X_val,   min(len(X_val),   batch_size))

    best_loss, best_state, best_epoch = float("inf"), None, 0
    pbar = tqdm(range(epochs), desc=f"Train [{label}]", leave=False)
    for epoch in pbar:
        model.train()
        for x_start in loader:
            t, _ = sampler.sample(x_start.shape[0], device)
            noise = torch.randn_like(x_start)
            x_t   = gd.q_sample(x_start.float(), t, noise=noise)
            loss  = ((noise - model(x_t, gd._scale_timesteps(t)))**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        if epoch % 50 == 0 and epoch > 200:
            model.eval()
            with torch.no_grad():
                vl = [((noise - model(gd.q_sample(xs.float(), t, noise=(noise := torch.randn_like(xs))),
                                      gd._scale_timesteps(t)))**2).mean().item()
                      for xs in val_loader
                      for t, _ in [sampler.sample(xs.shape[0], device)]]
                val_loss = float(np.mean(vl))
            if val_loss < best_loss:
                best_loss, best_state, best_epoch = val_loss, deepcopy(model.state_dict()), epoch
            pbar.set_postfix(val=f"{val_loss:.4f}", best=best_epoch)
            if epoch - best_epoch > patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [{label}] ep={epoch}  best_val={best_loss:.4f}  best_ep={best_epoch}")
    return model, mu, std


def diffusion_hessian(model, gd, X_np, mu, std, t_step):
    model.eval()
    X_norm   = torch.FloatTensor((X_np - mu) / std)
    n, d     = X_norm.shape
    s_ab     = float(gd.sqrt_one_minus_alphas_cumprod[t_step])
    t_single = torch.ones(1, dtype=torch.long) * t_step
    t_scaled = gd._scale_timesteps(t_single)

    def score_fn(x):
        return (-model(x, t_scaled) / s_ab).squeeze(0)  # (d,)

    # vmap vectorises over batch; jacrev computes full d×d Jacobian per sample
    # input:  (n, 1, d)  →  output: (n, d, 1, d)  →  squeeze dim 2 → (n, d, d)
    H_full = vmap(jacrev(score_fn))(X_norm.unsqueeze(1)).squeeze(2).detach().numpy()
    return np.array([H_full[:, j, j] for j in range(d)]).T, H_full


# =========================================================================
# Data
# =========================================================================
seed = 42
rng  = np.random.default_rng(seed)
n    = 1000

X_full  = rng.standard_normal((n, 2))
mask    = X_full[:, 0] + X_full[:, 1] > 0
X_trunc = X_full[mask]
X_eval_full  = X_full[:500]
X_eval_trunc = X_trunc[:500]

print(f"Full: n={len(X_full)}  Truncated: n={len(X_trunc)}")

# =========================================================================
# Fit all non-neural estimators
# =========================================================================
stein = SteinHessian()
kde   = KDEHessian()
dsm   = DSMHessian(sigma=0.3, seed=seed)
flow  = GaussianFlowHessian()

_, Hf_stein_full  = stein.fit(X_eval_full);   _, Hf_stein_trunc  = stein.fit(X_eval_trunc)
_, Hf_kde_full    = kde.fit(X_eval_full);     _, Hf_kde_trunc    = kde.fit(X_eval_trunc)
_, Hf_dsm_full    = dsm.fit(X_eval_full);     _, Hf_dsm_trunc    = dsm.fit(X_eval_trunc)
_, Hf_flow_full   = flow.fit(X_eval_full);    _, Hf_flow_trunc   = flow.fit(X_eval_trunc)

# =========================================================================
# Train and evaluate diffusion models (t=50, t=80 only)
# =========================================================================
gd = build_diffusion(n_steps=100)
print("\nTraining on full data…")
model_full,  mu_f, std_f = train_score_model(X_full,  gd, label="full")
print("Training on truncated data…")
model_trunc, mu_t, std_t = train_score_model(X_trunc, gd, label="trunc")

print("\nComputing diffusion Jacobians…")
t_steps = [50, 80]
results_diff = {}
for t in t_steps:
    _, Hf_f = diffusion_hessian(model_full,  gd, X_eval_full,  mu_f, std_f, t)
    _, Hf_t = diffusion_hessian(model_trunc, gd, X_eval_trunc, mu_t, std_t, t)
    results_diff[t] = dict(Hf_full=Hf_f, Hf_trunc=Hf_t)

# =========================================================================
# Collect all results for unified summary
# =========================================================================
all_methods = [
    ("True",           None,           None),
    ("Stein",          Hf_stein_full,  Hf_stein_trunc),
    ("KDE",            Hf_kde_full,    Hf_kde_trunc),
    ("DSM (σ=0.3)",    Hf_dsm_full,    Hf_dsm_trunc),
    ("Gaussian flow",  Hf_flow_full,   Hf_flow_trunc),
    ("Diffusion t=50", results_diff[50]["Hf_full"], results_diff[50]["Hf_trunc"]),
    ("Diffusion t=80", results_diff[80]["Hf_full"], results_diff[80]["Hf_trunc"]),
]

print("\n" + "="*72)
print("Diagonal Hessian — variance comparison (true var = 0)")
print("="*72)
print(f"\n  {'Method':<20} {'var(H[0,0]) full':>17} {'var(H[0,0]) trunc':>18} {'ratio':>7}")
print(f"  {'-'*65}")
for name, Hf, Ht in all_methods:
    if Hf is None:
        print(f"  {'True (N(0,I))':<20} {'0':>17} {'0':>18}")
        continue
    vf = Hf[:, 0, 0].var()
    vt = Ht[:, 0, 0].var()
    ratio = vt / vf if vf > 1e-12 else float("inf")
    print(f"  {name:<20} {vf:>17.4f} {vt:>18.4f} {ratio:>7.2f}x")

print(f"\n  {'Method':<20} {'mean H[0,0] full':>17} {'mean H[0,0] trunc':>18}")
print(f"  {'-'*58}")
for name, Hf, Ht in all_methods:
    if Hf is None:
        print(f"  {'True (N(0,I))':<20} {-1:>17.4f} {-1:>18.4f}")
        continue
    print(f"  {name:<20} {Hf[:,0,0].mean():>17.4f} {Ht[:,0,0].mean():>18.4f}")

# =========================================================================
# Plot: bar chart (variance) + per-method H[0,0] histograms
# =========================================================================
method_labels = ["Stein", "KDE", "DSM", "Flow", "Diff\nt=50", "Diff\nt=80"]
Hf_list = [Hf_stein_full,  Hf_kde_full,  Hf_dsm_full,  Hf_flow_full,
           results_diff[50]["Hf_full"],  results_diff[80]["Hf_full"]]
Ht_list = [Hf_stein_trunc, Hf_kde_trunc, Hf_dsm_trunc, Hf_flow_trunc,
           results_diff[50]["Hf_trunc"], results_diff[80]["Hf_trunc"]]

colors_full  = ["#4C72B0", "#4C72B0", "#4C72B0", "#4C72B0", "#8172B3", "#8172B3"]
colors_trunc = ["#DD8452", "#DD8452", "#DD8452", "#DD8452", "#C44E52", "#C44E52"]

fig, axes = plt.subplots(2, len(Hf_list), figsize=(14, 9))

# Row 0: H[0,0] full data
# Row 1: H[0,0] truncated
# Row 2: shared bar chart of variance
for col, (lbl, Hf, Ht, cf, ct) in enumerate(
        zip(method_labels, Hf_list, Ht_list, colors_full, colors_trunc)):

    for row, (H, color, tag) in enumerate([(Hf, cf, "full"), (Ht, ct, "trunc")]):
        ax = axes[row, col]
        v = H[:, 0, 0]
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        ax.hist(v, bins=50, range=(lo, hi), color=color, edgecolor="none", alpha=0.85)
        ax.axvline(-1, color="k", ls="--", lw=1.2)
        ax.set_title(f"{lbl}\n({tag})", fontsize=8)
        ax.set_xlabel("H[0,0]", fontsize=7)
        ax.annotate(f"μ={v.mean():+.2f}\nσ²={v.var():.3f}",
                    xy=(0.04, 0.96), xycoords="axes fraction", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

# Row 2: variance bar chart for all methods
ax_bar = fig.add_axes([0.08, 0.04, 0.88, 0.22])
x = np.arange(len(Hf_list))
vf_vals = [H[:, 0, 0].var() for H in Hf_list]
vt_vals = [H[:, 0, 0].var() for H in Ht_list]
ax_bar.bar(x - 0.2, vf_vals, width=0.38, color="#4C72B0", alpha=0.85, label="full data")
ax_bar.bar(x + 0.2, vt_vals, width=0.38, color="#DD8452", alpha=0.85, label="truncated")
ax_bar.axhline(0, color="k", ls="--", lw=1, label="true = 0")
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(["Stein", "KDE", "DSM (σ=0.3)", "Gaussian flow",
                         "Diffusion t=50", "Diffusion t=80"], fontsize=8)
ax_bar.set_ylabel("var(H[0,0])")
ax_bar.set_title("Diagonal Hessian variance — full vs truncated", fontsize=9)
ax_bar.legend(fontsize=8)

# Tighten main rows
for ax in axes.flat:
    ax.tick_params(labelsize=7)
fig.subplots_adjust(top=0.95, bottom=0.30, hspace=0.55, wspace=0.4)
fig.suptitle("Hessian estimator comparison: X1,X2~N(0,1), selection X1+X2>0\n"
             "True H[0,0]=−1, true var=0", fontsize=10)

plt.savefig("score_estim_test.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to score_estim_test.png")
