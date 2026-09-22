"""Finite-graph Newman-Ziff bond occupation with binomial conversion.

Q_n is the LCC fraction after exactly n occupied edges. The canonical mean is
sum Binom(m,n;p) Q_n and its derivative is m E[Q_(N+1)-Q_N], N~Binom(m-1,p).
The finite-graph diagnostic is the first maximum of this derivative on a
declared full-domain grid; it is not an infinite-network phase transition.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from numba import njit
from scipy.sparse import csr_matrix
from scipy.stats import binom

METHOD_VERSION = "canonical_v2_exact_endpoint"


@njit(cache=True)
def count_curve(n_nodes: int, edges: np.ndarray, order: np.ndarray) -> np.ndarray:
    parent = np.arange(n_nodes)
    size = np.ones(n_nodes, dtype=np.int64)
    largest = 1
    result = np.empty(len(edges) + 1, dtype=np.float64)
    result[0] = 1.0 / n_nodes
    for step, index in enumerate(order):
        u, v = edges[index, 0], edges[index, 1]
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        if u != v:
            if size[u] < size[v]:
                u, v = v, u
            parent[v] = u
            size[u] += size[v]
            largest = max(largest, size[u])
        result[step + 1] = largest / n_nodes
    return result


def sample_counts(n_nodes: int, edges: np.ndarray, n_orders: int, seed: int,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    if n_nodes < 1 or len(edges) < 1 or n_orders < 1:
        raise ValueError("A nonempty edge set and positive nodes/orders are required")
    if edges.min() < 0 or edges.max() >= n_nodes:
        raise ValueError("Edge labels must index the declared nodes")
    rng = np.random.default_rng(seed) if rng is None else rng
    curves = np.empty((n_orders, len(edges) + 1))
    for i in range(n_orders):
        curves[i] = count_curve(n_nodes, edges, rng.permutation(len(edges)))
    return curves


def binomial_kernel(m: int, grid: np.ndarray, tail: float = 1e-14) -> csr_matrix:
    if m < 0 or np.any((grid < 0) | (grid > 1)):
        raise ValueError("Invalid binomial trial count or probability grid")
    values, columns, offsets = [], [], [0]
    for p in grid:
        if p == 0 or p == 1 or m == 0:
            k = np.array([m if p == 1 else 0], dtype=int)
            weight = np.ones(1)
        else:
            low = max(0, int(binom.ppf(tail, m, p)))
            high = min(m, int(binom.ppf(1 - tail, m, p)))
            k = np.arange(low, high + 1)
            weight = binom.pmf(k, m, p)
            if abs(weight.sum() - 1) > 1e-10:
                raise ArithmeticError("Binomial kernel lost excessive probability mass")
        values.extend(weight.tolist())
        columns.extend(k.tolist())
        offsets.append(len(values))
    return csr_matrix((values, columns, offsets), shape=(len(grid), m + 1))


def canonical_curves(counts: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = counts.shape[1] - 1
    means = binomial_kernel(m, grid) @ counts.T
    derivatives = m * (binomial_kernel(m - 1, grid) @ np.diff(counts, axis=1).T)
    return np.asarray(means), np.asarray(derivatives)


def peak(grid: np.ndarray, derivative: np.ndarray) -> float:
    highest = derivative.max()
    indices = np.flatnonzero(np.isclose(derivative, highest, rtol=1e-12, atol=1e-12))
    return float(grid[indices[0]])


def endpoint_slope(n_nodes: int, edges: np.ndarray) -> tuple[float, int]:
    """Exact Q'(1)=sum_e [S1(G)-S1(G-e)], using bridge subtree sizes."""
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(edges)
    components = sorted(nx.connected_components(g), key=len, reverse=True)
    largest = len(components[0])
    second = len(components[1]) if len(components) > 1 else 0
    if second == largest:
        return 0.0, 0
    subgraph = g.subgraph(components[0])
    tree = list(nx.dfs_edges(subgraph, source=min(components[0])))
    sizes = {node: 1 for node in components[0]}
    parent = {}
    for u, v in reversed(tree):
        sizes[u] += sizes[v]
        parent[v] = u
    loss, bridges = 0, 0
    for u, v in nx.bridges(subgraph):
        child = u if parent.get(u) == v else v
        count = sizes[child]
        loss += largest - max(count, largest-count, second)
        bridges += 1
    return loss / n_nodes, bridges


def estimate(n_nodes: int, edges: np.ndarray, n_orders: int = 2048,
             n_bootstrap: int = 1000, seed: int = 20260905,
             step: float = 0.0025) -> tuple[dict, dict]:
    grid = np.linspace(0, 1, int(round(1 / step)) + 1)
    m = len(edges)
    q_kernel, d_kernel = binomial_kernel(m, grid), binomial_kernel(m-1, grid)
    curves = np.empty((len(grid), n_orders))
    derivative = np.empty_like(curves)
    micro = np.empty_like(curves)
    rng_orders = np.random.default_rng(seed)
    for start in range(0, n_orders, 64):
        stop = min(start + 64, n_orders)
        counts = sample_counts(n_nodes, edges, stop-start, seed, rng=rng_orders)
        curves[:, start:stop] = q_kernel @ counts.T
        derivative[:, start:stop] = m * (d_kernel @ np.diff(counts, axis=1).T)
        micro[:, start:stop] = counts[:, np.rint(grid * m).astype(int)].T
    exact_endpoint, n_bridges = endpoint_slope(n_nodes, edges)
    sampled_endpoint = float(derivative[-1].mean())
    derivative[-1, :] = exact_endpoint
    mean_d = derivative.mean(axis=1)
    pc = peak(grid, mean_d)
    rng = np.random.default_rng(seed + 7919)
    weights = rng.multinomial(n_orders, np.full(n_orders, 1 / n_orders), n_bootstrap)
    boot_d = derivative @ (weights.T / n_orders)
    draws = np.array([peak(grid, boot_d[:, i]) for i in range(n_bootstrap)])
    micro_pc = peak(grid, np.gradient(micro.mean(axis=1), grid))
    result = {
        "pc": pc, "pc_low": float(np.quantile(draws, .025)),
        "pc_high": float(np.quantile(draws, .975)),
        "pc_fixed_count_same_orders": micro_pc,
        "canonical_minus_fixed": pc - micro_pc,
        "point_endpoint": pc in (0.0, 1.0),
        "bootstrap_endpoint_count": int(np.sum((draws == 0) | (draws == 1))),
        "p_step": step, "n_orders": n_orders, "n_bootstrap": n_bootstrap,
        "mean_curve_p0": float(curves[0].mean()),
        "mean_curve_p1": float(curves[-1].mean()),
        "exact_endpoint_slope": exact_endpoint,
        "sampled_endpoint_slope": sampled_endpoint,
        "largest_component_bridges": n_bridges,
        "method_version": METHOD_VERSION,
    }
    for size in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        if size <= n_orders:
            result[f"pc_first_{size}_orders"] = peak(grid, derivative[:, :size].mean(axis=1))
    return result, {"grid": grid, "curves": curves, "derivative": derivative,
                    "bootstrap_pc": draws, "micro_mean": micro.mean(axis=1)}


def strict_flips(predicted_fc: np.ndarray, observed_fc: np.ndarray,
                 targets: np.ndarray, atol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    predicted_margin = np.asarray(predicted_fc)[:, None] - targets
    observed_margin = np.asarray(observed_fc)[:, None] - targets
    unsafe = (predicted_margin > atol) & (observed_margin < -atol)
    safe = (predicted_margin < -atol) & (observed_margin > atol)
    return unsafe, safe
