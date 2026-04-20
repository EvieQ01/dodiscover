"""Data generator for the 4-node fork graph: X1 -> X3 <- X2, X3 -> X4.

Supports linear/nonlinear mechanisms and multiple noise distributions.
"""

import numpy as np
import pandas as pd
import networkx as nx


def generate_fork_data(
    n_samples=3000,
    w13=1.2,
    w23=0.5,
    w34=1.3,
    mechanism="nonlinear",
    noise="gaussian",
    noise_scale=1.0,
    seed=42,
):
    """Generate data from the fork SCM: X1 -> X3 <- X2, X3 -> X4.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    w13, w23, w34 : float
        Edge weights for X1->X3, X2->X3, X3->X4.
    mechanism : {"linear", "nonlinear"}
        "linear": X3 = w13*X1 + w23*X2 + N3, X4 = w34*X3 + N4
        "nonlinear": X3 = sinh(w13*X1) + sinh(w23*X2) + N3, X4 = tanh(w34*X3) + N4
    noise : {"gaussian", "logistic", "laplace", "exponential"}
        Noise distribution for all nodes.
    noise_scale : float
        Scale parameter for the noise distribution.
    seed : int or None
        Random seed.

    Returns
    -------
    data : pd.DataFrame
        Columns X1, X2, X3, X4.
    true_graph : nx.DiGraph
        Ground-truth DAG.
    info : dict
        Metadata (weights, mechanism, noise type, etc.).
    """
    rng = np.random.default_rng(seed)

    noise_samplers = {
        "gaussian": lambda n: rng.normal(0, noise_scale, n),
        "logistic": lambda n: rng.logistic(0, noise_scale, n),
        "laplace": lambda n: rng.laplace(0, noise_scale, n),
        "exponential": lambda n: rng.exponential(noise_scale, n) - noise_scale,
    }
    if noise not in noise_samplers:
        raise ValueError(f"Unknown noise type '{noise}'. Choose from {list(noise_samplers)}")
    sample_noise = noise_samplers[noise]

    N1 = sample_noise(n_samples)
    N2 = sample_noise(n_samples)
    N3 = sample_noise(n_samples)
    N4 = sample_noise(n_samples)

    X1 = N1
    X2 = N2

    if mechanism == "linear":
        X3 = w13 * X1 + w23 * X2 + N3
        X4 = w34 * X3 + N4
    elif mechanism == "nonlinear":
        X3 = np.sinh(w13 * X1) + np.sinh(w23 * X2) + N3
        X4 = np.tanh(w34 * X3) + N4
    else:
        raise ValueError(f"Unknown mechanism '{mechanism}'. Choose 'linear' or 'nonlinear'")

    data = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "X4": X4})

    true_graph = nx.DiGraph([("X1", "X3"), ("X2", "X3"), ("X3", "X4")])

    info = {
        "n_samples": n_samples,
        "weights": {"w13": w13, "w23": w23, "w34": w34},
        "mechanism": mechanism,
        "noise": noise,
        "noise_scale": noise_scale,
        "seed": seed,
        "true_edges": set(true_graph.edges()),
        "true_skeleton": {frozenset(e) for e in true_graph.to_undirected().edges()},
        "true_sinks": {n for n in true_graph.nodes() if true_graph.out_degree(n) == 0},
        "true_sources": {n for n in true_graph.nodes() if true_graph.in_degree(n) == 0},
    }

    return data, true_graph, info