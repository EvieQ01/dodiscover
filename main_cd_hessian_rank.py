"""
Causal Discovery via Hessian Eigenspectrum Rank
================================================

Implements Algorithm 1 from "Sink Identification via Hessian Eigenspectrum —
A Residual-Free Method for Score-Based Causal Discovery under Deterministic
Selection."

The key idea: for each node X_j, form the gradient outer-product matrix

    M_j = E[ H_j(X) H_j(X)^T ]

where H_j(x) is the j-th row of the Hessian of log p(x).  The effective
rank of M_j is strictly smaller for sink nodes than for non-sinks.
Sinks are identified iteratively (peeling), then the resulting topological
order is pruned with CAM regression to yield the final DAG.

Pipeline
--------
1. Generate ANM data  (nonlinear tanh mechanisms, Gaussian noise)
2. Train a diffusion score model  (DiffAN)
3. Iteratively identify sinks via Hessian eigenspectrum with DiffAN-style
   masking and multi-step voting  (Algorithm 1)
4. Prune with CAM regression
5. Compare with SCORE and PC baselines

Graph under study
-----------------
    X1 --> X3 <-- X2
            |
            v
           X4

True edges  : X1->X3, X2->X3, X3->X4
True sinks  : {X4}
True sources: {X1, X2}
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DiffAN"))

import numpy as np
import pandas as pd
import networkx as nx
import torch
from functorch import vmap, jacrev

from diffan.diffan import DiffAN as DiffANCore
from dodiscover.toporder._base import CAMPruning
from dodiscover.toporder.score import SCORE
from dodiscover.toporder.utils import full_dag
from dodiscover.ci import FisherZCITest
from dodiscover import PC, make_context
from data_generation_fork import generate_fork_data
from CCA_tools import EigRankTest

# ======================================================================
# 1.  Reproducibility & configuration
# ======================================================================

seed = 42
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

N_SAMPLES = 3000
DIFFAN_EPOCHS = 1500
N_EVAL = 500  # samples per Hessian evaluation (per t-step)

# ======================================================================
# 2.  ANM data generation
# ======================================================================
# Nonlinear mechanisms (sinh/tanh) + configurable noise distribution.
# Non-Gaussianity ensures g'(N) is non-constant, giving a rank gap
# in the gradient outer-product matrix  M_j  (Proposition 1):
#   Sink  : rank(M) ≤ |PA|+1   (only own-noise directions)
#   Non-sink: rank(M) ≤ |PA|+|CH|+1  (additional child-noise directions)
# The gap is ≥ |CH| ≥ 1  (Table 1, nonlinear + non-Gaussian row).

MECHANISM = "linear"  # "linear" or "nonlinear"
NOISE_TYPE = "laplace"  # "gaussian", "logistic", "laplace", "exponential"

data, true_graph, gen_info = generate_fork_data(
    n_samples=N_SAMPLES,
    w13=1.2,
    w23=0.5,
    w34=1.3,
    mechanism=MECHANISM,
    noise=NOISE_TYPE,
    seed=seed,
)
cols = list(data.columns)
n_nodes = len(cols)
w13, w23, w34 = gen_info["weights"]["w13"], gen_info["weights"]["w23"], gen_info["weights"]["w34"]
true_edges = gen_info["true_edges"]
true_skeleton = gen_info["true_skeleton"]
true_sinks = gen_info["true_sinks"]
true_sources = gen_info["true_sources"]

print("=" * 65)
print("Causal Discovery via Hessian Eigenspectrum Rank")
print("=" * 65)
print(f"Graph : X1 -> X3 <- X2,  X3 -> X4")
print(f"Weights: w13={w13:.2f}  w23={w23:.2f}  w34={w34:.2f}")
print(f"Mechanism: {MECHANISM}  |  Noise: {NOISE_TYPE}  |  n={N_SAMPLES}")
print("=" * 65)

# ======================================================================
# 3.  Train diffusion score model (DiffAN)
# ======================================================================

print(f"\n[1/5] Training diffusion score model ({DIFFAN_EPOCHS} epochs) ...")

X_np = data.to_numpy(dtype=float)
mu = X_np.mean(0, keepdims=True)
std = X_np.std(0, keepdims=True)
X_norm = (X_np - mu) / std

core = DiffANCore(n_nodes, epochs=DIFFAN_EPOCHS)
X_tensor = torch.FloatTensor(X_norm).to(core.device)
core.train_score(X_tensor)


# ======================================================================
# 4.  Hessian computation helper
# ======================================================================
def compute_hessian_at_step(core, X_eval, t_step, active_nodes=None):
    """Compute per-sample Hessian of score in Z-space at a given diffusion step.

    When *active_nodes* is given, inactive nodes are masked to zero in the
    input (DiffAN masking strategy) and the output Jacobian is restricted to
    the active rows/columns.

    Returns
    -------
    H : ndarray, shape (n_eval, |active|, |active|)
    """
    core.model.eval()
    gd = core.gaussian_diffusion
    s_ab = float(gd.sqrt_one_minus_alphas_cumprod[t_step])
    t_s = torch.ones(1, dtype=torch.long).to(core.device) * t_step
    t_scaled = gd._scale_timesteps(t_s)
    d = X_eval.shape[1]

    if active_nodes is None:
        active_nodes = list(range(d))

    # Snapshot the active index list for the closure
    _active = list(active_nodes)

    def score_fn(z):
        """Score restricted to active nodes."""
        return (-core.model(z, t_scaled) / s_ab).squeeze(0)[_active]

    # Mask inactive nodes to zero (DiffAN masking)
    mask = torch.zeros(d, device=core.device)
    mask[_active] = 1.0
    X_masked = X_eval * mask

    H_full = vmap(jacrev(score_fn))(X_masked.unsqueeze(1)).squeeze(2)
    # H_full shape: (n_eval, |active|, d) — restrict columns to active
    H_active = H_full[:, :, _active].detach().cpu().numpy()
    return H_active  # (n_eval, |active|, |active|)


# ======================================================================
# 5.  Algorithm 1 — Hessian eigenspectrum sink identification
# ======================================================================
# At each peeling step:
#   a) compute the Hessian at several diffusion time-steps (multi-step voting,
#      same strategy as DiffAN's topological_ordering)
#   b) form M_j and compute effective rank for each remaining node
#   c) identify the sink as argmin(eff_rank), averaged over time-steps

print(f"\n[2/5] Computing Hessians at multiple diffusion steps ...")

# Compute full Hessian (no masking) at multiple t-steps for averaging
n_steps = core.n_steps
# T_STEPS = [3, 5, 10, n_steps // 3]
T_STEPS = [5, 10]

X_eval_tensor = X_tensor[:N_EVAL]

H_list = []  # list of (N_EVAL, d, d) Hessians at each t-step
for t_step in T_STEPS:
    H_t = compute_hessian_at_step(core, X_eval_tensor, t_step)  # full, no masking
    H_list.append(H_t)

# Also compute Stein Hessian for theoretical comparison (in Z-space, same as diffusion)
print("  Computing Stein Hessian (subsample n=500) ...")
from dodiscover.toporder._base import SteinMixin

STEIN_N = 500
stein = SteinMixin()
H_stein = stein.hessian(X_norm[:STEIN_N], eta_G=0.001, eta_H=0.001)  # (STEIN_N, d, d)
# Keep in Z-space for consistency with diffusion Hessian
scale_matrix = std.squeeze()[:, None] * std.squeeze()[None, :]


# ======================================================================
# 5a. Eigenspectrum ordering function
# ======================================================================

def eigenspectrum_peeling(
    H_list_or_single, node_names, label="", center=False, rank_method="effective"
):
    """Algorithm 1: iterative eigenspectrum sink identification.

    Parameters
    ----------
    H_list_or_single : list of ndarray or single ndarray, each (n, d, d)
        Per-sample Hessians. If a list, ranks are averaged across them.
    node_names : list of str
    center : bool
        If True, use Cov(H_j) instead of E[H_j H_j^T].  Removes mean
        inflation that makes effective rank ≈ 1 for all nodes.
    rank_method : {"effective", "chi2"}
        "effective": r_j = (sum lambda)^2 / sum(lambda^2)
        "chi2": Bartlett sequential test via EigRankTest (from CCA_tools)

    Returns
    -------
    order_idx : list of int   (root → leaf)
    diagnostics : list of dict
    """
    if isinstance(H_list_or_single, np.ndarray):
        H_all = [H_list_or_single]
    else:
        H_all = H_list_or_single

    d = len(node_names)
    remaining = list(range(d))
    sink_order = []
    diagnostics = []

    while len(remaining) > 1:
        avg_ranks = {}
        all_eigs = {}

        for j in remaining:
            ranks_j = []
            for H in H_all:
                H_j = H[:, j, :][:, remaining]  # (n_eval, |remaining|)

                if center:
                    H_j = H_j - H_j.mean(axis=0)
                    # scaling by variance
                    H_j = H_j / (H_j.std(axis=0, keepdims=True) + 1e-15)

                n = H_j.shape[0]
                M_j = (H_j.T @ H_j) / n
                eigs = np.sort(np.maximum(np.linalg.eigvalsh(M_j), 0))[::-1]

                if rank_method == "chi2":
                    rt = EigRankTest(H_j, center=False)  # already centered above if needed
                    ranks_j.append(float(rt.estimate_rank(alpha=0.05)))
                else:  # effective rank
                    s1 = eigs.sum()
                    s2 = (eigs**2).sum()
                    ranks_j.append(s1**2 / s2 if s2 > 1e-15 else 0.0)

                all_eigs[j] = eigs

            avg_ranks[j] = np.mean(ranks_j)

        sink = min(remaining, key=lambda j: avg_ranks[j])

        diagnostics.append(
            {
                "step": len(sink_order),
                "remaining": [node_names[i] for i in remaining],
                "eff_ranks": {node_names[j]: avg_ranks[j] for j in remaining},
                "eigenvalues": {node_names[j]: all_eigs[j] for j in remaining},
                "sink": node_names[sink],
            }
        )

        sink_order.append(sink)
        remaining.remove(sink)

    sink_order.append(remaining[0])
    sink_order.reverse()
    return sink_order, diagnostics


# ======================================================================
# 5b. Run Algorithm 1 with diffusion Hessian (multi-step averaged)
# ======================================================================

print(f"\n[3/5] Running Hessian Eigenspectrum Algorithm (Algorithm 1) ...")

order_idx, diagnostics = eigenspectrum_peeling(H_list, cols, label="Diffusion")
order_names = [cols[i] for i in order_idx]

# Also run with Stein Hessian for comparison
order_idx_stein, diag_stein = eigenspectrum_peeling(H_stein, cols, label="Stein")
order_names_stein = [cols[i] for i in order_idx_stein]

# Build fully connected DAG from the order, then CAM-prune
def build_pruned_dag(order_indices, X_data, column_names):
    """Build a DAG from a topological order + CAM pruning."""
    d = len(column_names)
    A_dense = full_dag(order_indices)
    pruner = CAMPruning(alpha=0.001)
    G_inc = nx.DiGraph()
    G_inc.add_nodes_from(range(d))
    G_exc = nx.DiGraph()
    G_exc.add_nodes_from(range(d))
    A_pruned = pruner.prune(X_data, A_dense, G_inc, G_exc)
    G = nx.DiGraph()
    G.add_nodes_from(column_names)
    for i in range(d):
        for j in range(d):
            if A_pruned[i, j] == 1:
                G.add_edge(column_names[i], column_names[j])
    return G


G_hessian = build_pruned_dag(order_idx, X_np, cols)
G_hessian_stein = build_pruned_dag(order_idx_stein, X_np, cols)

# ======================================================================
# 6.  SCORE algorithm  (Stein-estimator baseline)
# ======================================================================

print(f"\n[3/5] Running SCORE algorithm ...")

context = make_context().variables(data=data).build()
score_alg = SCORE(alpha=0.01)
score_alg.learn_graph(data, context)
G_score = score_alg.graph_

# ======================================================================
# 7.  PC algorithm  (constraint-based baseline)
# ======================================================================

print(f"\n[4/5] Running PC algorithm ...")

ci_test = FisherZCITest()
pc_alg = PC(ci_estimator=ci_test, alpha=0.05)
pc_alg.learn_graph(data, context)
G_pc_cpdag = pc_alg.graph_

# Convert CPDAG → skeleton + directed edges for evaluation
pc_directed_edges = set()
pc_skeleton = set()
try:
    pc_dg = G_pc_cpdag.sub_directed_graph()
    for u, v in pc_dg.edges():
        pc_directed_edges.add((u, v))
        pc_skeleton.add(frozenset([u, v]))
except Exception:
    pass
try:
    pc_ug = G_pc_cpdag.sub_undirected_graph()
    for u, v in pc_ug.edges():
        pc_skeleton.add(frozenset([u, v]))
except Exception:
    pass

# Build a DAG-like view of PC output for uniform evaluation
G_pc_dag = nx.DiGraph()
G_pc_dag.add_nodes_from(cols)
for u, v in pc_directed_edges:
    G_pc_dag.add_edge(u, v)
# Undirected edges → both directions (represents orientation uncertainty)
for edge in pc_skeleton - {frozenset(e) for e in pc_directed_edges}:
    u, v = sorted(edge)
    if (u, v) not in pc_directed_edges and (v, u) not in pc_directed_edges:
        G_pc_dag.add_edge(u, v)
        G_pc_dag.add_edge(v, u)


# ======================================================================
# 8.  Evaluation helpers
# ======================================================================


def evaluate_dag(name, G_pred):
    """Evaluate a predicted DAG against the ground truth."""
    pred_edges = set(G_pred.edges())
    pred_skeleton = {frozenset(e) for e in G_pred.to_undirected().edges()}

    # Skeleton
    skel_tp = len(true_skeleton & pred_skeleton)
    skel_fp = len(pred_skeleton - true_skeleton)
    skel_fn = len(true_skeleton - pred_skeleton)
    prec = skel_tp / max(skel_tp + skel_fp, 1)
    rec = skel_tp / max(skel_tp + skel_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    # Orientation
    n_correct = len(true_edges & pred_edges)
    n_reversed = len({(v, u) for u, v in true_edges} & pred_edges)

    # SHD = missing + extra + reversed
    shd = skel_fp + skel_fn + n_reversed

    return {
        "name": name,
        "pred_edges": pred_edges,
        "skel_tp": skel_tp,
        "skel_fp": skel_fp,
        "skel_fn": skel_fn,
        "skel_f1": f1,
        "orient_correct": n_correct,
        "orient_reversed": n_reversed,
        "shd": shd,
    }


# ======================================================================
# 9.  Diagnostic: full-Hessian analysis at a single t-step
# ======================================================================
# Also compute the full (non-iterative) Hessian for diagnostic comparison
# with SCORE's diagonal-variance criterion.

print(f"\n[5/5] Computing diagnostic Hessian at t=10 ...")

H_full_diag = compute_hessian_at_step(core, X_eval_tensor, 10)
# Rescale Z → X for the diagonal variance diagnostic
scale_diag = std.squeeze()  # (d,)
H_X_diag = np.zeros_like(H_full_diag)
for i in range(n_nodes):
    for k in range(n_nodes):
        H_X_diag[:, i, k] = H_full_diag[:, i, k] / (scale_diag[i] * scale_diag[k])


# ======================================================================
# 9b. Analytical Hessian of log p(x) for comparison
# ======================================================================
# For Gaussian noise (sigma=1):
#   log p_N(n) = -n²/2 + C,  so  g(n) = d/dn log p_N = -n,  g'(n) = -1
#
# Nonlinear SCM:
#   f13(x1) = sinh(w13*x1),  f23(x2) = sinh(w23*x2),  f34(x3) = tanh(w34*x3)
# Linear SCM:
#   f13(x1) = w13*x1,  f23(x2) = w23*x2,  f34(x3) = w34*x3
#
# Residuals:  r3 = x3 - f13(x1) - f23(x2),  r4 = x4 - f34(x3)
#
# Hessian diagonal (Gaussian noise, sigma=1):
#   H[j,j] = -1 - sum_over_children[ (f'_child)² ] + sum_over_children[ r_child * f''_child ]
#   H[4,4] = -1  (sink — no children, always constant)

print("\n[5b/5] Computing ANALYTICAL Hessian for comparison ...")

X_eval_np = X_np[:N_EVAL]
x1, x2, x3, x4 = X_eval_np[:, 0], X_eval_np[:, 1], X_eval_np[:, 2], X_eval_np[:, 3]

H_analytic_diag = np.zeros((N_EVAL, n_nodes))
H_analytic_full = np.zeros((N_EVAL, n_nodes, n_nodes))

if MECHANISM == "nonlinear":
    # f13 = sinh(w13*x1): f13' = w13*cosh, f13'' = w13²*sinh
    # f23 = sinh(w23*x2): f23' = w23*cosh, f23'' = w23²*sinh
    # f34 = tanh(w34*x3): f34' = w34*sech², f34'' = -2*w34²*sech²*tanh
    r3 = x3 - np.sinh(w13 * x1) - np.sinh(w23 * x2)
    r4 = x4 - np.tanh(w34 * x3)

    f13_p = w13 * np.cosh(w13 * x1)
    f13_pp = w13**2 * np.sinh(w13 * x1)
    f23_p = w23 * np.cosh(w23 * x2)
    f23_pp = w23**2 * np.sinh(w23 * x2)
    sech2 = (1.0 / np.cosh(w34 * x3)) ** 2
    f34_p = w34 * sech2
    f34_pp = -2 * w34**2 * sech2 * np.tanh(w34 * x3)

    H_analytic_diag[:, 0] = -1 - f13_p**2 + r3 * f13_pp
    H_analytic_diag[:, 1] = -1 - f23_p**2 + r3 * f23_pp
    H_analytic_diag[:, 2] = -1 - f34_p**2 + r4 * f34_pp
    H_analytic_diag[:, 3] = -1.0  # SINK

    # Off-diagonal: from Eq.7, ∂²logp/∂x1∂x2 = -f13'·f23' (co-parents of X3)
    H_analytic_full[:, 0, 1] = -f13_p * f23_p
    H_analytic_full[:, 0, 2] = f13_p
    H_analytic_full[:, 1, 2] = f23_p
    H_analytic_full[:, 2, 3] = f34_p

elif MECHANISM == "linear":
    # f13 = w13*x1: f13' = w13, f13'' = 0
    # f23 = w23*x2: f23' = w23, f23'' = 0
    # f34 = w34*x3: f34' = w34, f34'' = 0
    H_analytic_diag[:, 0] = -1 - w13**2
    H_analytic_diag[:, 1] = -1 - w23**2
    H_analytic_diag[:, 2] = -1 - w34**2
    H_analytic_diag[:, 3] = -1.0  # SINK

    H_analytic_full[:, 0, 1] = -w13 * w23  # co-parent off-diagonal (constant)
    H_analytic_full[:, 0, 2] = w13
    H_analytic_full[:, 1, 2] = w23
    H_analytic_full[:, 2, 3] = w34

# Fill symmetric + diagonal
for i in range(n_nodes):
    H_analytic_full[:, i, i] = H_analytic_diag[:, i]
H_analytic_full[:, 1, 0] = H_analytic_full[:, 0, 1]
H_analytic_full[:, 2, 0] = H_analytic_full[:, 0, 2]
H_analytic_full[:, 2, 1] = H_analytic_full[:, 1, 2]
H_analytic_full[:, 3, 2] = H_analytic_full[:, 2, 3]

# Run eigenspectrum peeling on the analytical Hessian — compare methods
# (a) Original: non-centered + effective rank (broken for this setting)
order_idx_analytic_orig, diag_analytic_orig = eigenspectrum_peeling(
    H_analytic_full, cols, label="Analytic-orig", center=False, rank_method="effective"
)
# (b) Centered + effective rank (removes mean inflation)
order_idx_analytic_ctr, diag_analytic_ctr = eigenspectrum_peeling(
    H_analytic_full, cols, label="Analytic-ctr", center=True, rank_method="effective"
)
# (c) Centered + chi2 rank test (proper statistical rank)
order_idx_analytic_chi2, diag_analytic_chi2 = eigenspectrum_peeling(
    H_analytic_full, cols, label="Analytic-chi2", center=True, rank_method="chi2"
)

# Use chi2 method as the primary analytical result
order_idx_analytic = order_idx_analytic_chi2
diag_analytic = diag_analytic_chi2
order_names_analytic = [cols[i] for i in order_idx_analytic]

G_hessian_analytic = build_pruned_dag(order_idx_analytic, X_np, cols)

results = [
    evaluate_dag("HR-Diffusion", G_hessian),
    evaluate_dag("HR-Stein", G_hessian_stein),
    evaluate_dag("HR-Analytic", G_hessian_analytic),
    evaluate_dag("SCORE", G_score),
    evaluate_dag("PC", G_pc_dag),
]

# ======================================================================
# 10.  Print results
# ======================================================================

print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

# --- 10a. Eigenspectrum diagnostics (Algorithm 1 peeling steps) ---
print("\n-- Eigenspectrum Diagnostics (Algorithm 1 peeling steps) --")
for step in diagnostics:
    print(f"\n  Step {step['step']}: remaining nodes = {step['remaining']}")
    for node in step["remaining"]:
        er = step["eff_ranks"][node]
        eigs = step["eigenvalues"][node]
        top3 = ", ".join(f"{e:.4f}" for e in eigs[:3])
        tag = " <-- SINK" if node == step["sink"] else ""
        print(f"    {node:<4}  eff_rank={er:.3f}  top eigs=[{top3}]{tag}")
    print(f"    -> Identified sink: {step['sink']}")

print(f"\n  Topological order (root -> leaf): {order_names}")

print("\n-- Eigenspectrum with Stein Hessian (theoretical reference) --")
for step in diag_stein:
    print(f"\n  Step {step['step']}: remaining nodes = {step['remaining']}")
    for node in step["remaining"]:
        er = step["eff_ranks"][node]
        eigs = step["eigenvalues"][node]
        top3 = ", ".join(f"{e:.4f}" for e in eigs[:3])
        tag = " <-- SINK" if node == step["sink"] else ""
        print(f"    {node:<4}  eff_rank={er:.3f}  top eigs=[{top3}]{tag}")
    print(f"    -> Identified sink: {step['sink']}")
print(f"\n  Topological order (Stein): {order_names_stein}")

# --- 10a'. Analytical Hessian: compare rank estimation methods ---
print("\n-- ANALYTICAL Hessian: Rank Method Comparison (step 0) --")
print(f"  Theory (nonlinear+Gaussian, Table 1): sink rank <= |PA|+1, non-sink rank <= |PA|+|CH|+1")
print(f"  X4 sink: rank <= 2 | X3 non-sink: rank <= 4 | X1,X2 non-sink: rank <= 2*")
print(f"  (*paper bound |PA|+|CH|+1=2 is not tight for fork graphs; true rank is 3)")
print()

# Get step-0 diagnostics from each method
methods = [
    ("Non-centered + eff_rank", diag_analytic_orig),
    ("Centered + eff_rank", diag_analytic_ctr),
    ("Centered + chi2_rank", diag_analytic_chi2),
]
header = f"  {'Node':<6}"
for mname, _ in methods:
    header += f" {mname:>24}"
header += f" {'True rank':>10}"
print(header)
print(f"  {'-' * (6 + 25 * len(methods) + 12)}")

# Compute analytical true ranks for reference
true_ranks = {}
for j in range(n_nodes):
    H_j = H_analytic_full[:, j, :] - H_analytic_full[:, j, :].mean(axis=0)
    eigs = np.linalg.eigvalsh(H_j.T @ H_j / H_j.shape[0])
    true_ranks[cols[j]] = int(np.sum(eigs / eigs.max() > 1e-6))

for node in cols:
    row = f"  {node:<6}"
    for mname, diag in methods:
        step0 = diag[0]
        val = step0["eff_ranks"][node]
        tag = " *" if node == step0["sink"] else ""
        row += f" {val:>22.3f}{tag}"
    row += f" {true_ranks[node]:>10}"
    print(row)

print()
for mname, diag in methods:
    order = list(reversed([diag[i]["sink"] for i in range(len(diag))]))
    order.insert(0, [n for n in cols if n not in order][0])
    print(f"  {mname}: order = {order}, sink = {diag[0]['sink']}")
print(f"  True sinks: {true_sinks}  |  True sources: {true_sources}")

# --- 10b. Hessian diagonal variance (SCORE-style diagnostic) ---
# Compare: analytical (clean) vs diffusion (noised at t=10) vs Stein (kernel)
print("\n-- Hessian Diagonal Variance: Analytical vs Diffusion vs Stein --")
print(f"  {'Node':<6} {'Analytical':>12} {'Diffusion':>12} {'Stein':>12} {'Role':>10}")
print(f"  {'-' * 56}")
diag_vars = []
diag_vars_analytic = []
diag_vars_stein = []
for i, col in enumerate(cols):
    v_diff = H_X_diag[:, i, i].var()
    v_anal = H_analytic_diag[:, i].var()
    v_stein = H_stein[:, i, i].var()
    diag_vars.append(v_diff)
    diag_vars_analytic.append(v_anal)
    diag_vars_stein.append(v_stein)
    role = "SINK" if col in true_sinks else ("SOURCE" if col in true_sources else "")
    print(f"  {col:<6} {v_anal:>12.6f} {v_diff:>12.6f} {v_stein:>12.6f} {role:>10}")
leaf_by_var = cols[int(np.argmin(diag_vars))]
leaf_by_anal = cols[int(np.argmin(diag_vars_analytic))]
leaf_by_stein = cols[int(np.argmin(diag_vars_stein))]
print(f"  argmin(Var)  Analytical: {leaf_by_anal}  |  Diffusion: {leaf_by_var}  |  Stein: {leaf_by_stein}")
print(f"  True sink: {list(true_sinks)[0]}")
print()
print("  NOTE: Analytically, Var(H[4,4])=0 and Var(H[2,2])=0 (both constant = -1).")
print("  The diffusion Hessian is computed at t=10 from the NOISED distribution,")
print("  not the clean data — so the sink property (constant diagonal) is lost.")

# --- 10c. Three-way comparison table ---
print("\n-- Comparison: HessianRank  vs  SCORE  vs  PC --")
print(f"  True edges   : {true_edges}")
print(f"  True skeleton: {true_skeleton}\n")

header = (
    f"  {'Method':<14} {'Skel TP':>7} {'Skel FP':>7} {'Skel FN':>7} "
    f"{'F1':>6} {'Orient':>7} {'Rev':>4} {'SHD':>4}"
)
print(header)
print(f"  {'-' * 60}")
for r in results:
    orient_str = f"{r['orient_correct']}/{len(true_edges)}"
    print(
        f"  {r['name']:<14} {r['skel_tp']:>7} {r['skel_fp']:>7} {r['skel_fn']:>7} "
        f"{r['skel_f1']:>6.3f} {orient_str:>7} {r['orient_reversed']:>4} {r['shd']:>4}"
    )

print(f"\n  Predicted edges:")
for r in results:
    print(f"    {r['name']:<14}: {r['pred_edges']}")

# ======================================================================
# 11.  Diagnostic plot — eigenvalue spectra of M_j
# ======================================================================

import matplotlib.pyplot as plt

# Use the first peeling step (all 4 nodes active) for the spectrum plot
step0 = diagnostics[0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Left panel: eigenvalue spectra ---
ax = axes[0]
colors = {"X1": "tab:blue", "X2": "tab:orange", "X3": "tab:green", "X4": "tab:red"}
for node in step0["remaining"]:
    eigs = step0["eigenvalues"][node]
    marker = "s" if node in true_sinks else "o"
    label = f"{node} (sink)" if node in true_sinks else node
    ax.semilogy(
        range(1, len(eigs) + 1),
        eigs + 1e-15,
        "-" + marker,
        color=colors[node],
        label=label,
        markersize=8,
        linewidth=2,
    )
ax.set_xlabel("Eigenvalue index k")
ax.set_ylabel("lambda_k  (log scale)")
ax.set_title("Eigenvalue spectrum of M_j\n(gradient outer-product matrix, step 0)")
ax.legend()
ax.set_xticks(range(1, n_nodes + 1))
ax.grid(True, alpha=0.3)

# --- Right panel: effective rank bar chart ---
ax = axes[1]
nodes_sorted = sorted(step0["eff_ranks"].keys())
eff_ranks_vals = [step0["eff_ranks"][n] for n in nodes_sorted]
bar_colors = [colors[n] for n in nodes_sorted]
hatches = ["///" if n in true_sinks else "" for n in nodes_sorted]
bars = ax.bar(
    nodes_sorted, eff_ranks_vals, color=bar_colors, edgecolor="black", linewidth=0.8
)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
ax.set_ylabel("Effective rank  r_j")
ax.set_title("Effective rank per node\n(sink = argmin, hatched = true sink)")
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="rank = 1")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

fig.suptitle(
    "Hessian Eigenspectrum Sink Identification - Diagnostic",
    fontsize=12,
    y=1.02,
)
fig.tight_layout()
plt.savefig("hessian_rank_diagnostic.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Plot saved to hessian_rank_diagnostic.png")

# ======================================================================
# 12.  DAG comparison plot
# ======================================================================

fig, axes = plt.subplots(1, 6, figsize=(26, 4.5))
pos = {"X1": (0, 1), "X2": (2, 1), "X3": (1, 0.5), "X4": (1, 0)}

titles = ["Ground Truth", "HR-Diffusion", "HR-Stein", "HR-Analytic", "SCORE", "PC (CPDAG)"]
graphs = [true_graph, G_hessian, G_hessian_stein, G_hessian_analytic, G_score, G_pc_dag]

for ax, title, G in zip(axes, titles, graphs):
    edge_colors = []
    for u, v in G.edges():
        if (u, v) in true_edges:
            edge_colors.append("green")
        elif (v, u) in true_edges:
            edge_colors.append("orange")  # reversed
        else:
            edge_colors.append("red")  # spurious

    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        with_labels=True,
        node_color="lightblue",
        node_size=800,
        font_size=11,
        font_weight="bold",
        edge_color=edge_colors if G != true_graph else "black",
        arrows=True,
        arrowsize=20,
        width=2,
        connectionstyle="arc3,rad=0.1",
    )
    ax.set_title(title, fontsize=11)
    ax.axis("off")

# Legend
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="green", lw=2, label="Correct"),
    Line2D([0], [0], color="orange", lw=2, label="Reversed"),
    Line2D([0], [0], color="red", lw=2, label="Spurious"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10)

fig.suptitle("DAG Comparison", fontsize=13, y=1.01)
fig.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("dag_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Plot saved to dag_comparison.png")
