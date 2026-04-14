"""
Test DiffAN-based causal discovery on the graph:

    X1 --> X3 <-- X2
            |
            v
           X4

True edges  : X1->X3, X2->X3, X3->X4
True leaf   : X4
True sources: X1, X2

SEM (nonlinear, additive Gaussian noise, σ=1):
    X1 = N1
    X2 = N2
    X3 = tanh(w13·X1) + tanh(w23·X2) + N3
    X4 = tanh(w34·X3) + N4   ← leaf

Nonlinearity is necessary for a non-trivial closed-form Hessian.
For a linear Gaussian SEM, H(log p) is constant → var = 0 everywhere,
making the argmin criterion degenerate.  With tanh functions:

    H[X4,X4] = −1  (constant) → Var = 0   ← leaf correctly identified
    H[X3,X3] varies through x3            → Var > 0
    H[X1,X1] varies through x1            → Var > 0
    H[X2,X2] varies through x2            → Var > 0

Closed-form formula (change-of-variables):
    H(log p(x)) = Jᵀ · diag(h(nⱼ)) · J  +  Σⱼ sⱼ(nⱼ) · H_{nⱼ}(x)
    where  nⱼ = xⱼ − fⱼ(pa(xⱼ))   (noise realisations)
           J[j,k] = ∂nⱼ/∂xₖ        (Jacobian of noise w.r.t. data)
           h(n) = −1  for N(0,1)
           s(n) = −n  for N(0,1)
           H_{nⱼ}[i,k] = ∂²nⱼ/∂xᵢ∂xₖ  (second-order structural correction)

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
from functorch import vmap, jacrev

from diffan.diffan import DiffAN as DiffANCore
from diffan.utils import full_DAG as diffan_full_dag
from dodiscover.toporder._base import CAMPruning

# -------------------------------------------------------------------------
# Reproducibility and data generation
# -------------------------------------------------------------------------
seed = 42
rng  = np.random.default_rng(seed)
torch.manual_seed(seed)
n_samples = 3000


def random_weight():
    magnitude = rng.uniform(0.8, 1.8)
    sign      = rng.choice([-1.0, 1.0])
    return float(sign * magnitude)


w13 = random_weight()   # X1 -> X3
w23 = random_weight() * 1.5  # X2 -> X3
w34 = random_weight() * 1.5  # X3 -> X4

print("=" * 60)
print("Graph:  X1 --> X3 <-- X2,  X3 --> X4  (nonlinear SEM)")
print(f"Weights: X1->X3={w13:.3f}  X2->X3={w23:.3f}  X3->X4={w34:.3f}")
print("=" * 60)

X1 = rng.standard_normal(n_samples)
X2 = rng.standard_normal(n_samples)
X3 = np.tanh(w13 * X1) + np.tanh(w23 * X2) + rng.standard_normal(n_samples)
X4 = np.tanh(w34 * X3)  + 2 * rng.standard_normal(n_samples)  # tanh: non-saturating → var(H[X3]) stays large

df_full = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "X4": X4})
cols    = list(df_full.columns)          # ["X1","X2","X3","X4"]

# Ground-truth graph
true_graph    = nx.DiGraph([("X1", "X3"), ("X2", "X3"), ("X3", "X4")])
true_leaves   = {n for n in true_graph.nodes() if true_graph.out_degree(n) == 0}
true_skeleton = {frozenset(e) for e in true_graph.to_undirected().edges()}
true_edges    = set(true_graph.edges())
TRUE_LEAF_IDX = cols.index(list(true_leaves)[0])  # integer index of X4


# -------------------------------------------------------------------------
# Closed-form Hessian  H(log p(x))
#
# Jacobian J[j,k] = ∂nⱼ/∂xₖ  (lower-triangular in causal order):
#
#   J = [[ 1,       0,        0,  0 ],
#        [ 0,       1,        0,  0 ],
#        [-w13/ch²(w13·x0), -w23/ch²(w23·x1),  1,  0 ],
#        [ 0,       0,  -w34/ch²(w34·x2),  1 ]]
#
# Second-order structural corrections (∂²nⱼ/∂xᵢ²):
#   ∂²n3/∂x1² = 2·w13²·sech²(w13·x1)·tanh(w13·x1)
#   ∂²n3/∂x2² = 2·w23²·sech²(w23·x2)·tanh(w23·x2)
#   ∂²n4/∂x3² = 2·w34²·sech²(w34·x3)·tanh(w34·x3)
#
# Predictions:
#   Var(H[X4,X4]) = 0   (constant = -1)  ← leaf
#   Var(H[X3,X3]) > 0   (varies with x3 through X4's nonlinear dep)
#   Var(H[X1,X1]) > 0, Var(H[X2,X2]) > 0
# -------------------------------------------------------------------------

def closed_form_hessian(data: pd.DataFrame) -> np.ndarray:
    """Per-sample exact Hessian of log p(x); shape (n, 4, 4)."""
    X           = data.to_numpy(dtype=float)
    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    n0 = x0
    n1 = x1
    n2 = x2 - np.tanh(w13 * x0) - np.tanh(w23 * x1)
    n3 = x3 - np.tanh(w34 * x2)                         # X4 = tanh(w34·X3) + N4

    d = 4
    n = len(X)
    J = np.zeros((n, d, d))
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = 1.0
    J[:, 2, 0] = -w13 / np.cosh(w13 * x0) ** 2         # ∂n3/∂x1 = -w13·sech²(w13·x1)
    J[:, 2, 1] = -w23 / np.cosh(w23 * x1) ** 2         # ∂n3/∂x2 = -w23·sech²(w23·x2)
    J[:, 2, 2] = 1.0
    J[:, 3, 2] = -w34 * np.cosh(w34 * x2) ** (-2)      # ∂n4/∂x3 = -w34·sech²(w34·x3)  [tanh derivative]
    J[:, 3, 3] = 1.0

    h  = np.full((n, d), -1.0)                          # Gaussian h = -1/σ²
    H1 = np.einsum('nj, nji, njk -> nik', h, J, J)      # Jᵀ diag(h) J

    s  = -np.stack([n0, n1, n2, n3], axis=1)            # score = -nⱼ
    C  = np.zeros((n, d, d))
    C[:, 0, 0] += s[:, 2] * 2 * w13**2 / np.cosh(w13 * x0)**2 * np.tanh(w13 * x0)
    C[:, 1, 1] += s[:, 2] * 2 * w23**2 / np.cosh(w23 * x1)**2 * np.tanh(w23 * x1)
    # ∂²n4/∂x3² = w34²·sin(w34·x3)   [from -w34·cos derivative]
    C[:, 2, 2] += s[:, 3] * 2*w34**2 / np.cosh(w34*x2)**2 * np.tanh(w34*x2)
    # C[:, 3, 3] = 0  (leaf has no children)

    return H1 + C   # (n, 4, 4)


# -------------------------------------------------------------------------
# DiffAN wrapper: score model training + topological ordering + CAM pruning
# -------------------------------------------------------------------------

def run_diffan(data: pd.DataFrame, epochs: int = 1500, label: str = ""):
    """Train a DiffAN score model; return object with .graph_ .order_ .core ."""
    n_nodes = len(data.columns)
    X_np    = data.to_numpy(dtype=float)
    mu      = X_np.mean(0, keepdims=True)
    std     = X_np.std(0,  keepdims=True)
    X_norm  = (X_np - mu) / std

    core = DiffANCore(n_nodes, epochs=epochs)
    X_t  = torch.FloatTensor(X_norm).to(core.device)
    core.train_score(X_t)
    order = core.topological_ordering(X_t)   # root → leaf

    A_dense  = diffan_full_dag(order)
    pruner   = CAMPruning(alpha=0.001)
    G_incl   = nx.DiGraph(); G_incl.add_nodes_from(range(n_nodes))
    G_excl   = nx.DiGraph(); G_excl.add_nodes_from(range(n_nodes))
    A_pruned = pruner.prune(X_norm, A_dense, G_incl, G_excl)

    G = nx.DiGraph()
    G.add_nodes_from(data.columns)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if A_pruned[i, j] == 1:
                G.add_edge(data.columns[i], data.columns[j])

    class _Res: pass
    res        = _Res()
    res.graph_ = G
    res.order_ = order
    res.core   = core
    res.mu     = mu
    res.std    = std
    return res


# -------------------------------------------------------------------------
# Diffusion Hessian  (full matrix, rescaled to X-space)
# -------------------------------------------------------------------------

def diffusion_hessian_X(core, mu, std, X_np, t_step, n_eval=300):
    """
    Per-sample Hessian of log p(x) from diffusion score, in X-space.

    Model operates on Z = (X−μ)/σ.  Rescale back:
        H_X[i,k] = H_Z[i,k] / (std_i · std_k)

    Returns ndarray of shape (n_eval, d, d).
    """
    core.model.eval()
    gd       = core.gaussian_diffusion
    s_ab     = float(gd.sqrt_one_minus_alphas_cumprod[t_step])
    t_s      = torch.ones(1, dtype=torch.long).to(core.device) * t_step
    t_scaled = gd._scale_timesteps(t_s)

    def score_fn(z):
        return (-core.model(z, t_scaled) / s_ab).squeeze(0)

    Z_eval = torch.FloatTensor((X_np[:n_eval] - mu) / std).to(core.device)
    H_Z    = vmap(jacrev(score_fn))(Z_eval.unsqueeze(1)).squeeze(2).detach().cpu().numpy()

    scale = std.squeeze()[:, None] * std.squeeze()[None, :]   # (d, d)
    return H_Z / scale[None]                                   # (n_eval, d, d)


# -------------------------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------------------------

def evaluate(label: str, model) -> None:
    pred_graph   = model.graph_
    order_labels = [df_full.columns[i] for i in model.order_]
    pred_leaves  = {n for n in pred_graph.nodes() if pred_graph.out_degree(n) == 0}
    pred_skel    = {frozenset(e) for e in pred_graph.to_undirected().edges()}

    skel_tp = len(true_skeleton & pred_skel)
    skel_fp = len(pred_skel - true_skeleton)
    skel_fn = len(true_skeleton - pred_skel)

    pred_edges = set(pred_graph.edges())
    n_correct  = len(true_edges & pred_edges)
    n_reversed = len({(v, u) for u, v in true_edges} & pred_edges)

    print(f"\n  [DiffAN — {label}]")
    print(f"  Order       : {order_labels}")
    print(f"  Pred leaves : {pred_leaves}  (true: {true_leaves})  correct={pred_leaves == true_leaves}")
    print(f"  Skeleton    : TP={skel_tp}  FP={skel_fp}  FN={skel_fn}")
    print(f"  Orientation : {n_correct}/{len(true_edges)} correct, {n_reversed} reversed")
    print(f"  Pred edges  : {pred_edges}")


def print_hessian_table(title, H_cf, H_diff, n_ev):
    """Compare closed-form vs diffusion Hessian diagonal across nodes."""
    n_ev = min(H_cf.shape[0], H_diff.shape[0], n_ev)
    print(f"\n  ── {title} ──")
    print(f"  {'Node':<6}  {'mean cf':>9}  {'mean diff':>10}  "
          f"{'var cf':>10}  {'var diff':>10}  {'corr':>7}  {'role'}")
    print(f"  {'-'*72}")
    var_cf   = []
    var_diff = []
    for i, col in enumerate(cols):
        he   = H_cf[:n_ev, i, i]
        hd   = H_diff[:n_ev, i, i]
        corr = float(np.corrcoef(he, hd)[0, 1]) if he.var() > 1e-10 else float("nan")
        role = "← LEAF" if col in true_leaves else ""
        print(f"  {col:<6}  {he.mean():>9.4f}  {hd.mean():>10.4f}  "
              f"{he.var():>10.6f}  {hd.var():>10.6f}  {corr:>7.4f}  {role}")
        var_cf.append(he.var())
        var_diff.append(hd.var())

    leaf_cf   = cols[int(np.argmin(var_cf))]
    leaf_diff = cols[int(np.argmin(var_diff))]
    ok_cf     = leaf_cf   in true_leaves
    ok_diff   = leaf_diff in true_leaves
    print(f"  argmin:  cf → {leaf_cf} ({'✓' if ok_cf else '✗'})  |  "
          f"diffusion → {leaf_diff} ({'✓' if ok_diff else '✗'})  |  "
          f"true → {list(true_leaves)[0]}")


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

S         = df_full["X1"] + df_full["X4"]
trunc_lo, trunc_hi = 0.0, 0.5
s_lo, s_hi = S.quantile(trunc_lo), S.quantile(trunc_hi)

df_trunc  = df_full[(S > s_lo) & (S < s_hi)].reset_index(drop=True)
print(f"\n  Samples after truncation : {len(df_trunc)} / {n_samples}")

margin    = 0.05
s_span    = s_hi - s_lo
S_trunc   = df_trunc["X1"] + df_trunc["X4"]
df_interior = df_trunc[
    (S_trunc > s_lo + margin * s_span) & (S_trunc < s_hi - margin * s_span)
].reset_index(drop=True)
print(f"  Samples in interior      : {len(df_interior)}")

model_trunc = run_diffan(df_interior, label="interior")
evaluate("interior (selection bias)", model_trunc)

# -------------------------------------------------------------------------
# Diagnostic: closed-form vs diffusion Hessian diagonal variance
#
# Closed-form prediction:
#   Var(H[X4,X4]) = 0  (constant −1) → argmin correctly identifies X4
#   Var(H[X3,X3]), Var(H[X1,X1]), Var(H[X2,X2]) > 0
#
# Under truncation: H(log P(X|S=1)) = H(log P(X)) in the interior,
# so the closed-form variance ordering is preserved.
# Question: does the diffusion model also preserve the leaf argmin?
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Diagnostic: closed-form vs diffusion Hessian (leaf = argmin var)")
print("=" * 60)

T_STEP = 10
N_EVAL = 1000
# breakpoint()
H_cf_full  = closed_form_hessian(df_full)
H_cf_trunc = closed_form_hessian(df_interior)

print(f"\n  Computing diffusion Hessians at t={T_STEP} on {N_EVAL} samples…")
H_diff_full  = diffusion_hessian_X(
    model_full.core,  model_full.mu,  model_full.std,
    df_full.to_numpy(dtype=float),     T_STEP, N_EVAL,
)
H_diff_trunc = diffusion_hessian_X(
    model_trunc.core, model_trunc.mu, model_trunc.std,
    df_interior.to_numpy(dtype=float), T_STEP, N_EVAL,
)

print_hessian_table("Full data",               H_cf_full,  H_diff_full,  N_EVAL)
print_hessian_table("Truncated (S=X1+X4∈int)", H_cf_trunc, H_diff_trunc, N_EVAL)

# Closed-form variance shift due to truncation
print(f"\n  ── Closed-form variance shift due to truncation ──")
print(f"  {'Node':<6}  {'var full':>10}  {'var trunc':>11}  {'ratio':>7}")
for i, col in enumerate(cols):
    vf = H_cf_full[:N_EVAL,  i, i].var()
    vt = H_cf_trunc[:N_EVAL, i, i].var()
    role = "← LEAF" if col in true_leaves else ""
    print(f"  {col:<6}  {vf:>10.6f}  {vt:>11.6f}  "
          f"{vt/max(vf, 1e-12):>7.3f}  {role}")

# -------------------------------------------------------------------------
# Plot: X3 distribution before and after truncation
# (X3 is the non-leaf parent of X4; its Hessian variance is under investigation)
# -------------------------------------------------------------------------
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].hist(df_full["X3"],     bins=60, color="steelblue",  edgecolor="white", lw=0.3)
axes[0].set_title("X3 — full data");     axes[0].set_xlabel("X3"); axes[0].set_ylabel("count")

axes[1].hist(df_trunc["X3"],    bins=60, color="darkorange",  edgecolor="white", lw=0.3)
axes[1].set_title(f"X3 — truncated\n(S ∈ [{trunc_lo},{trunc_hi}] quantiles)")
axes[1].set_xlabel("X3")

axes[2].hist(df_interior["X3"], bins=60, color="seagreen",   edgecolor="white", lw=0.3)
axes[2].set_title(f"X3 — interior\n(+{int(margin*100)}% boundary margin)")
axes[2].set_xlabel("X3")

for ax, n in zip(axes, [len(df_full), len(df_trunc), len(df_interior)]):
    ax.annotate(f"n = {n}", xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9)

fig.suptitle("X3 (non-leaf, parent of X4) distribution under selection on S = X1 + X4",
             fontsize=11, y=1.01)
fig.tight_layout()
plt.savefig("x3_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Plot saved to x3_distribution.png")
