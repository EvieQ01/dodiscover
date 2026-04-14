"""
Selection bias on Hessian estimation: closed-form vs diffusion model.

Truncation:  S = 1{X0 + X2 > 0}   (source + leaf)

  log P(X|S=1) = log P(X) + log 1_S(X) - log Z
  H(log P(X|S=1)) = H(log P(X)) + H(log 1_S(X))

  In the interior of {X0+X2>0}:  H(log 1_S) = 0
  ⟹  H(log P_trunc) = H(log P)  at every interior point

  The closed-form Hessian is therefore the SAME function evaluated at
  truncated samples. Differences in the variance of H_ii arise because
  the truncated distribution has a different sample geometry (shifted
  mean, covariance, etc.) even though the Hessian function itself is
  unchanged in the interior.

Two examples
────────────
  SEM 1 – LINEAR  + Logistic noise
    X0 = N0
    X1 = 1.5·X0 + N1
    X2 = 1.0·X0 + 0.9·X1 + N2        (leaf)

    Closed-form (no second-order correction for linear SEM):
      H = Jᵀ · diag(h(nⱼ)) · J
      h(n) = −sech²(n/2)/2   (logistic curvature)

    Analytical variance prediction:
      Var(H[0,0]) ∝ 1 + 1.5⁴ + 1.0⁴  ≈ 7.06  (2 children)
      Var(H[1,1]) ∝ 1 + 0.9⁴          ≈ 1.66
      Var(H[2,2]) ∝ 1                   = 1.00  ← leaf (argmin)

  SEM 2 – NONLINEAR + Gaussian noise
    X0 = N0
    X1 = sin(2·X0) + N1
    X2 = tanh(1.5·X0) + sin(0.8·X1) + N2   (leaf)

    Closed-form (second-order structural corrections present):
      H = Jᵀ · diag(−1/σ²) · J + Σⱼ sⱼ · H_{nⱼ}
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DiffAN"))

import numpy as np
import torch
from functorch import vmap, jacrev
from diffan.diffan import DiffAN

# ─────────────────────────────────────────────────────────────────────────────
# Global config
# ─────────────────────────────────────────────────────────────────────────────
seed      = 7
n_nodes   = 3
n_samples = 3000
T_STEP    = 50
N_EVAL    = 400
EPOCHS    = 1500
SIGMA     = 0.8   # Gaussian noise scale (SEM 2)

rng = np.random.default_rng(seed)
torch.manual_seed(seed)

# True DAG: X0→X1→X2, X0→X2
true_adj = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
TRUE_LEAF = 2

# Selection mask
def truncate(X):
    return X[(X[:, 0] + X[:, 2]) > 0]


# ─────────────────────────────────────────────────────────────────────────────
# SEM 1: Linear + Logistic noise
# ─────────────────────────────────────────────────────────────────────────────
W_01, W_02, W_12 = 1.5, 1.0, 0.9   # edge weights


def sim_linear(n, rng):
    N = rng.logistic(0, 1, (n, n_nodes))
    X = np.zeros((n, n_nodes))
    X[:, 0] = N[:, 0]
    X[:, 1] = W_01 * X[:, 0] + N[:, 1]
    X[:, 2] = W_02 * X[:, 0] + W_12 * X[:, 1] + N[:, 2]
    return X


def cf_hessian_linear(X):
    """
    Closed-form H(log p(x)) for linear logistic SEM.

    Linear SEM ⟹ H_{nⱼ} = 0, so:  H = Jᵀ · diag(h(nⱼ)) · J

    h(n) = d²/dn² log Logistic(n; 0,1) = −sech²(n/2) / 2
    """
    x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
    n0 = x0
    n1 = x1 - W_01 * x0
    n2 = x2 - W_02 * x0 - W_12 * x1

    J = np.zeros((len(X), n_nodes, n_nodes))
    J[:, 0, 0] = 1.0
    J[:, 1, 0] = -W_01;  J[:, 1, 1] = 1.0
    J[:, 2, 0] = -W_02;  J[:, 2, 1] = -W_12;  J[:, 2, 2] = 1.0

    def h(n):
        return -0.5 / np.cosh(n / 2) ** 2   # logistic scale=1

    h_all = np.stack([h(n0), h(n1), h(n2)], axis=1)   # (n, 3)
    return np.einsum('nj, nji, njk -> nik', h_all, J, J)   # (n, 3, 3)


# ─────────────────────────────────────────────────────────────────────────────
# SEM 2: Nonlinear + Gaussian noise
# ─────────────────────────────────────────────────────────────────────────────

def sim_nonlinear(n, rng):
    N = rng.standard_normal((n, n_nodes)) * SIGMA
    X = np.zeros((n, n_nodes))
    X[:, 0] = N[:, 0]
    X[:, 1] = np.sin(2.0 * X[:, 0]) + N[:, 1]
    X[:, 2] = np.tanh(1.5 * X[:, 0]) + np.sin(0.8 * X[:, 1]) + N[:, 2]
    return X


def cf_hessian_nonlinear(X):
    """
    Closed-form H(log p(x)) for nonlinear Gaussian SEM.

    H = Jᵀ · diag(−1/σ²) · J  +  Σⱼ sⱼ(nⱼ) · H_{nⱼ}(x)

    Second-order structural corrections (∂²nⱼ/∂xᵢ² ≠ 0):
      ∂²n₁/∂x₀²  = 4 sin(2x₀)
      ∂²n₂/∂x₀²  = 4.5 sech²(1.5x₀) tanh(1.5x₀)
      ∂²n₂/∂x₁²  = 0.64 sin(0.8x₁)
    """
    x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
    n0 = x0
    n1 = x1 - np.sin(2.0 * x0)
    n2 = x2 - np.tanh(1.5 * x0) - np.sin(0.8 * x1)

    J = np.zeros((len(X), n_nodes, n_nodes))
    J[:, 0, 0] = 1.0
    J[:, 1, 0] = -2 * np.cos(2 * x0);            J[:, 1, 1] = 1.0
    J[:, 2, 0] = -1.5 / np.cosh(1.5 * x0) ** 2
    J[:, 2, 1] = -0.8 * np.cos(0.8 * x1);        J[:, 2, 2] = 1.0

    h = np.full((len(X), n_nodes), -1.0 / SIGMA ** 2)
    H1 = np.einsum('nj, nji, njk -> nik', h, J, J)

    s = -np.stack([n0, n1, n2], axis=1) / SIGMA ** 2
    C = np.zeros((len(X), n_nodes, n_nodes))
    C[:, 0, 0] += s[:, 1] * 4 * np.sin(2 * x0)
    C[:, 0, 0] += s[:, 2] * 4.5 / np.cosh(1.5 * x0) ** 2 * np.tanh(1.5 * x0)
    C[:, 1, 1] += s[:, 2] * 0.64 * np.sin(0.8 * x1)
    return H1 + C


# ─────────────────────────────────────────────────────────────────────────────
# DiffAN helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_diffan(X, label=""):
    mu  = X.mean(0, keepdims=True)
    std = X.std(0,  keepdims=True)
    diffan = DiffAN(n_nodes, masking=True, residue=True, epochs=EPOCHS)
    X_t = torch.FloatTensor((X - mu) / std).to(diffan.device)
    diffan.train_score(X_t, fixed=EPOCHS)
    print(f"  [{label}] done — {len(X)} samples")
    return diffan, mu, std


def diff_hessian(diffan, X_np, mu, std, t_step, n_eval):
    """
    Per-sample Hessian from diffusion model, rescaled to X-space.
    Returns (n_eval, d, d).
    """
    diffan.model.eval()
    gd       = diffan.gaussian_diffusion
    s_ab     = float(gd.sqrt_one_minus_alphas_cumprod[t_step])
    t_s      = torch.ones(1, dtype=torch.long).to(diffan.device) * t_step
    t_scaled = gd._scale_timesteps(t_s)

    def score_fn(z):
        return (-diffan.model(z, t_scaled) / s_ab).squeeze(0)

    Z_eval = torch.FloatTensor((X_np[:n_eval] - mu) / std).to(diffan.device)
    H_Z = vmap(jacrev(score_fn))(Z_eval.unsqueeze(1)).squeeze(2).detach().cpu().numpy()

    scale = std.squeeze()[:, None] * std.squeeze()[None, :]   # (d, d)
    return H_Z / scale[None]   # (n_eval, d, d) in X-space


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table printer
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(title, X_eval, H_cf, H_diff):
    """
    Print per-node: mean and variance of diagonal Hessian for
    closed-form and diffusion, plus sample-level correlation.
    """
    n_ev = min(len(X_eval), H_cf.shape[0], H_diff.shape[0])
    print(f"\n  {title}")
    print(f"  n_eval={n_ev}")
    hdr = (f"  {'Node':<6}  {'mean (cf)':>10}  {'mean (diff)':>12}  "
           f"{'var (cf)':>10}  {'var (diff)':>11}  {'corr':>7}  {'role'}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i in range(n_nodes):
        he = H_cf[:n_ev, i, i]
        hd = H_diff[:n_ev, i, i]
        corr = float(np.corrcoef(he, hd)[0, 1]) if he.var() > 1e-12 else float("nan")
        role = "← LEAF" if i == TRUE_LEAF else ""
        print(f"  X{i:<5}  {he.mean():>10.4f}  {hd.mean():>12.4f}  "
              f"{he.var():>10.6f}  {hd.var():>11.6f}  {corr:>7.4f}  {role}")
    argmin_cf   = int(np.argmin([H_cf[:n_ev,  i, i].var() for i in range(n_nodes)]))
    argmin_diff = int(np.argmin([H_diff[:n_ev, i, i].var() for i in range(n_nodes)]))
    print(f"  argmin var:  closed-form → X{argmin_cf}  |  "
          f"diffusion → X{argmin_diff}  |  true leaf = X{TRUE_LEAF}")


# ─────────────────────────────────────────────────────────────────────────────
# Main experiments
# ─────────────────────────────────────────────────────────────────────────────

for sem_tag, sim_fn, cf_fn, sem_desc in [
    ("linear",    sim_linear,    cf_hessian_linear,
     "SEM 1: X0=N0, X1=1.5X0+N1, X2=X0+0.9X1+N2  (Logistic noise)"),
    ("nonlinear", sim_nonlinear, cf_hessian_nonlinear,
     "SEM 2: X0=N0, X1=sin(2X0)+N1, X2=tanh(1.5X0)+sin(0.8X1)+N2  (Gaussian noise)"),
]:
    print("\n" + "=" * 70)
    print(sem_desc)
    print("=" * 70)

    # ── simulate ────────────────────────────────────────────────────────────
    rng_local = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X_full  = sim_fn(n_samples, rng_local)
    X_trunc = truncate(X_full)
    print(f"\n  Full: n={len(X_full)}  |  Truncated (X0+X2>0): n={len(X_trunc)} "
          f"({100*len(X_trunc)/len(X_full):.0f}%)")

    # ── closed-form Hessians ────────────────────────────────────────────────
    H_cf_full  = cf_fn(X_full)
    H_cf_trunc = cf_fn(X_trunc)

    # ── train DiffAN ─────────────────────────────────────────────────────────
    print(f"\n  Training DiffAN on full data…")
    diffan_f, mu_f, std_f = train_diffan(X_full,  label=f"{sem_tag}_full")
    print(f"  Training DiffAN on truncated data…")
    diffan_t, mu_t, std_t = train_diffan(X_trunc, label=f"{sem_tag}_trunc")

    # ── diffusion Hessians ───────────────────────────────────────────────────
    H_df_full  = diff_hessian(diffan_f, X_full,  mu_f, std_f, T_STEP, N_EVAL)
    H_df_trunc = diff_hessian(diffan_t, X_trunc, mu_t, std_t, T_STEP, N_EVAL)

    # ── analytical variance prediction (linear only) ─────────────────────────
    if sem_tag == "linear":
        vN = np.stack([cf_hessian_linear(X_full[:, :])[:, i, i]
                       for i in range(n_nodes)], axis=1).var(0)
        v_h  = np.var(-0.5 / np.cosh(X_full[:, 0] / 2) ** 2)  # var(h(N))
        print(f"\n  Analytical variance prediction (proportional to var(h(N))):")
        print(f"    Var(H[0,0]) ∝ {1+W_01**4+W_02**4:.4f} × var(h(N))")
        print(f"    Var(H[1,1]) ∝ {1+W_12**4:.4f} × var(h(N))")
        print(f"    Var(H[2,2]) ∝ 1.0000 × var(h(N))")

    # ── print tables ─────────────────────────────────────────────────────────
    print(f"\n  ── Full data (t={T_STEP}) ──────────────────────────────────────")
    print_comparison("Closed-form  vs  Diffusion (trained on FULL)",
                     X_full, H_cf_full, H_df_full)

    print(f"\n  ── Truncated data  S = X0+X2 > 0  (t={T_STEP}) ─────────────────")
    print_comparison("Closed-form  vs  Diffusion (trained on TRUNCATED)",
                     X_trunc, H_cf_trunc, H_df_trunc)

    # ── variance shift table (cf only) ──────────────────────────────────────
    print(f"\n  ── Closed-form variance shift due to truncation ─────────────────")
    print(f"  {'Node':<6}  {'var full':>10}  {'var trunc':>11}  {'ratio':>7}  "
          f"{'ordering preserved?':>20}")
    v_full  = np.array([H_cf_full[:, i, i].var()  for i in range(n_nodes)])
    v_trunc = np.array([H_cf_trunc[:, i, i].var() for i in range(n_nodes)])
    order_full  = np.argsort(v_full)
    order_trunc = np.argsort(v_trunc)
    for i in range(n_nodes):
        role = "← LEAF" if i == TRUE_LEAF else ""
        print(f"  X{i:<5}  {v_full[i]:>10.6f}  {v_trunc[i]:>11.6f}  "
              f"{v_trunc[i]/max(v_full[i],1e-12):>7.3f}  {role}")
    preserved = (order_full[0] == TRUE_LEAF) and (order_trunc[0] == TRUE_LEAF)
    print(f"  Leaf argmin preserved: {preserved}  "
          f"(full→X{order_full[0]}, trunc→X{order_trunc[0]})")
