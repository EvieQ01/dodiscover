"""
Test DiffAN-based causal discovery on the graph:

    X1 --> X3 <-- X2
            |
            v
           X4

True edges  : X1->X3, X2->X3, X3->X4
True leaf   : X4
True sources: X1, X2

Score estimation uses a diffusion model (DiffMLP trained with denoising score
matching).  Topological ordering is based on the Jacobian of the learned score
at a fixed diffusion timestep (argmin diagonal variance = leaf).
CAM pruning uses dodiscover's LinearGAM-based implementation.

Two regimes
-----------
A. Full observational data
B. Selection truncation on S = X1 + X4, keeping only interior samples.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DiffAN"))

import networkx as nx
import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffan.diffan import DiffAN as DiffANCore
from diffan.gaussian_diffusion import (
    GaussianDiffusion, UniformSampler, get_named_beta_schedule,
    LossType, ModelMeanType, ModelVarType,
)
from diffan.nn import DiffMLP
from diffan.utils import full_DAG as diffan_full_dag

from dodiscover.toporder._base import CAMPruning

# -------------------------------------------------------------------------
# Reproducibility and data generation
# -------------------------------------------------------------------------
seed = 42
rng = np.random.default_rng(seed)
n_samples = 3000


def random_weight():
    magnitude = rng.uniform(0.8, 1.8)
    sign = rng.choice([-1.0, 1.0])
    return float(sign * magnitude)


w13 = random_weight()   # X1 -> X3
w23 = random_weight()   # X2 -> X3
w34 = random_weight()   # X3 -> X4

print("=" * 60)
print("Graph:  X1 --> X3 <-- X2,  X3 --> X4")
print(f"Weights: X1->X3={w13:.3f}  X2->X3={w23:.3f}  X3->X4={w34:.3f}")
print("=" * 60)

X1 = rng.standard_normal(n_samples)
X2 = rng.standard_normal(n_samples)
X3 = w13 * X1 + w23 * X2 + rng.standard_normal(n_samples)
X4 = w34 * X3 + rng.standard_normal(n_samples)

df_full = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "X4": X4})

# Ground-truth graph
true_graph = nx.DiGraph([("X1", "X3"), ("X2", "X3"), ("X3", "X4")])
true_leaves = {n for n in true_graph.nodes() if true_graph.out_degree(n) == 0}
true_skeleton = {frozenset(e) for e in true_graph.to_undirected().edges()}
true_edges = set(true_graph.edges())


# -------------------------------------------------------------------------
# DiffAN wrapper: score model training + topological ordering + CAM pruning
# -------------------------------------------------------------------------

def run_diffan(data: pd.DataFrame, epochs: int = 1500, label: str = ""):
    """Train a DiffAN score model and return an object with .graph_ and .order_.

    Topological ordering uses DiffAN's Jacobian-based leaf detection.
    Edge pruning uses dodiscover's CAMPruning (LinearGAM).
    """
    cols = list(data.columns)
    n_nodes = len(cols)
    X_np = data.to_numpy(dtype=float)

    # Z-score normalise (same as DiffAN.fit)
    mu  = X_np.mean(0, keepdims=True)
    std = X_np.std(0, keepdims=True)
    X_norm = (X_np - mu) / std

    # Train DiffAN score model
    core = DiffANCore(n_nodes, epochs=epochs)
    X_t = torch.FloatTensor(X_norm).to(core.device)
    core.train_score(X_t)

    # Topological ordering via Jacobian diagonal variance
    order = core.topological_ordering(X_t)   # list of ints, root → leaf

    # Dense adjacency matrix from ordering, then CAM pruning
    A_dense = diffan_full_dag(order)          # shape (n_nodes, n_nodes)
    pruner  = CAMPruning(alpha=0.001)
    G_incl  = nx.DiGraph()
    G_incl.add_nodes_from(range(n_nodes))
    G_excl  = nx.DiGraph()
    G_excl.add_nodes_from(range(n_nodes))
    A_pruned = pruner.prune(X_norm, A_dense, G_incl, G_excl)

    # Build named networkx graph
    G = nx.DiGraph()
    G.add_nodes_from(cols)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if A_pruned[i, j] == 1:
                G.add_edge(cols[i], cols[j])

    class _Result:
        pass

    res = _Result()
    res.graph_ = G
    res.order_ = order   # integer indices, root → leaf
    res.core   = core    # trained DiffANCore (has .model and .gaussian_diffusion)
    res.mu     = mu
    res.std    = std
    return res


# -------------------------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------------------------

def evaluate(label: str, model) -> None:
    pred_graph  = model.graph_
    order_labels = [df_full.columns[i] for i in model.order_]

    pred_leaves  = {n for n in pred_graph.nodes() if pred_graph.out_degree(n) == 0}
    pred_skeleton = {frozenset(e) for e in pred_graph.to_undirected().edges()}
    skel_tp = len(true_skeleton & pred_skeleton)
    skel_fp = len(pred_skeleton - true_skeleton)
    skel_fn = len(true_skeleton - pred_skeleton)

    pred_edges         = set(pred_graph.edges())
    correctly_oriented = len(true_edges & pred_edges)
    reversed_edges     = len({(v, u) for u, v in true_edges} & pred_edges)

    print(f"\n  [DiffAN — {label}]")
    print(f"  Order       : {order_labels}")
    print(f"  Pred leaves : {pred_leaves}  (true: {true_leaves})  correct={pred_leaves == true_leaves}")
    print(f"  Skeleton    : TP={skel_tp}  FP={skel_fp}  FN={skel_fn}")
    print(f"  Orientation : {correctly_oriented}/{len(true_edges)} correct, {reversed_edges} reversed")
    print(f"  Pred edges  : {pred_edges}")


# -------------------------------------------------------------------------
# Experiment A: Full observational data
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Experiment A: Full observational data")
print("=" * 60)

model_full = run_diffan(df_full, label="full")
evaluate("full data", model_full)


# -------------------------------------------------------------------------
# Experiment B: Selection truncation on S = X1 + X4
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Experiment B: Selection on S = X1 + X4  (interior trimming)")
print("=" * 60)

S = df_full["X1"] + df_full["X4"]
trunc_lo, trunc_hi = 0.15, 0.85
s_lo, s_hi = S.quantile(trunc_lo), S.quantile(trunc_hi)

mask_trunc = (S > s_lo) & (S < s_hi)
df_trunc   = df_full[mask_trunc].reset_index(drop=True)
print(f"\n  Samples after truncation : {len(df_trunc)} / {n_samples}")

margin  = 0.01
s_span  = s_hi - s_lo
S_trunc = df_trunc["X1"] + df_trunc["X4"]
mask_interior = (S_trunc > s_lo + margin * s_span) & (S_trunc < s_hi - margin * s_span)
df_interior   = df_trunc[mask_interior].reset_index(drop=True)
print(f"  Samples in interior      : {len(df_interior)}")

model_trunc = run_diffan(df_interior, label="interior")
evaluate("interior (selection bias)", model_trunc)


# -------------------------------------------------------------------------
# Diagnostic: Diffusion Hessian diagonal variances before and after selection
#
# SCORE / DiffAN identifies the leaf as argmin of var(H_diag) over samples.
# We compute the Jacobian of the diffusion score at timestep T_STEP and
# compare diagonal variances for the full-data model vs the interior model.
#
# The diffusion forward process blurs the hard boundary, so the Hessian
# estimate degrades much less under selection than the Stein estimator.
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Diagnostic: Diffusion Hessian diagonal variances  (leaf = argmin)")
print("=" * 60)

T_STEP  = 80    # diffusion timestep for Hessian evaluation
N_DIAG  = 200   # number of samples used (per-sample Jacobian is slow)
cols    = list(df_full.columns)


def diffusion_hessian_diag(core, mu, std, X_np, t_step, n_eval=200):
    """Compute per-sample diagonal Hessian of the diffusion score.

    score_k(x) = -eps_theta(x_t, t)_k / sqrt(1 - alpha_bar_t)
    H_kk(x)   = d score_k / d x_k  (diagonal of Jacobian w.r.t. input)

    Returns array of shape (n_eval, n_nodes).
    """
    X_eval  = X_np[:n_eval]
    device  = core.device
    core.model.eval()
    X_norm  = torch.FloatTensor((X_eval - mu) / std).to(device)
    n, d    = X_norm.shape
    s_ab    = float(core.gaussian_diffusion.sqrt_one_minus_alphas_cumprod[t_step])
    t_vec   = torch.ones(n, dtype=torch.long).to(device) * t_step

    H_diag = np.zeros((n, d))
    for i in range(n):
        x_i = X_norm[i:i+1].requires_grad_(True)
        eps  = core.model(x_i, core.gaussian_diffusion._scale_timesteps(t_vec[i:i+1]))
        score = -eps / s_ab
        for k in range(d):
            g = torch.autograd.grad(score[0, k], x_i, retain_graph=(k < d - 1))[0]
            H_diag[i, k] = g[0, k].item()
    return H_diag


print(f"\n  Using t={T_STEP}, evaluating on {N_DIAG} samples…")

H_diag_full = diffusion_hessian_diag(
    model_full.core,  model_full.mu,  model_full.std,
    df_full.to_numpy(dtype=float),     T_STEP, N_DIAG,
)
H_diag_int = diffusion_hessian_diag(
    model_trunc.core, model_trunc.mu, model_trunc.std,
    df_interior.to_numpy(dtype=float), T_STEP, N_DIAG,
)

var_full = H_diag_full.var(axis=0)
var_int  = H_diag_int.var(axis=0)

print(f"\n  {'Node':<6}  {'Var (full)':<14}  {'Var (interior)':<14}  {'ratio int/full'}")
print(f"  {'-'*55}")
for i, col in enumerate(cols):
    ratio = var_int[i] / var_full[i] if var_full[i] > 1e-12 else float("inf")
    print(f"  {col:<6}  {var_full[i]:<14.5f}  {var_int[i]:<14.5f}  {ratio:.3f}")

print(f"\n  Argmin (leaf) on full data     : {cols[int(np.argmin(var_full))]}")
print(f"  Argmin (leaf) on interior data : {cols[int(np.argmin(var_int))]}")
print(f"  True leaf                      : {list(true_leaves)[0]}")


# -------------------------------------------------------------------------
# Plot: X4 distribution before and after truncation
# -------------------------------------------------------------------------
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

axes[0].hist(df_full["X4"], bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
axes[0].set_title("X4 — full data")
axes[0].set_xlabel("X4")
axes[0].set_ylabel("count")

axes[1].hist(df_trunc["X4"], bins=60, color="darkorange", edgecolor="white", linewidth=0.3)
axes[1].set_title(f"X4 — after truncation\n(S ∈ [{trunc_lo},{trunc_hi}] quantiles)")
axes[1].set_xlabel("X4")

axes[2].hist(df_interior["X4"], bins=60, color="seagreen", edgecolor="white", linewidth=0.3)
axes[2].set_title(f"X4 — interior only\n(+{int(margin*100)}% boundary margin)")
axes[2].set_xlabel("X4")

for ax, n in zip(axes, [len(df_full), len(df_trunc), len(df_interior)]):
    ax.annotate(f"n = {n}", xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9)

fig.suptitle("Distribution of X4 (true leaf) under selection on S = X1 + X4",
             fontsize=11, y=1.01)
fig.tight_layout()
plt.savefig("x4_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Plot saved to x4_distribution.png")
