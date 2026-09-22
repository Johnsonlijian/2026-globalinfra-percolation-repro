"Portable, unchanged finite-graph helpers extracted from the verified computation."

from __future__ import annotations

import hashlib
import math
import random

import networkx as nx
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

HONEYCOMB = 1.0 - 2.0 * math.sin(math.pi / 18.0)
SQUARE = 0.5


def seed_for(name, base=20260905):
    return base + sum((i + 1) * ord(c) for i, c in enumerate(name.lower()))


def graph_hash(n, edges):
    ordered = np.sort(edges.astype("<i8"), axis=1)
    ordered = ordered[np.lexsort((ordered[:, 1], ordered[:, 0]))]
    return hashlib.sha256(np.array([n], dtype="<i8").tobytes() + ordered.tobytes()).hexdigest()


def spectral_threshold(n, edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    if nx.is_forest(graph):
        return float("inf"), 0.0, 0.0
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    m = len(edges)
    reverse = np.concatenate([np.arange(m, 2 * m), np.arange(m)])

    def multiply(x):
        return np.bincount(src, weights=x, minlength=n)[dst] - x[reverse]

    operator = LinearOperator((2 * m, 2 * m), matvec=multiply, dtype=float)
    value, vector = eigs(operator, k=1, which="LR", v0=np.ones(2 * m), tol=1e-9, maxiter=10000)
    radius = float(abs(value[0]))
    x = vector[:, 0]
    action = multiply(x.real) + 1j * multiply(x.imag)
    residual = float(np.linalg.norm(action - value[0] * x) / np.linalg.norm(x))
    if residual > 1e-6 or abs(value[0].imag) > 1e-7:
        raise ArithmeticError(f"Nonbacktracking eigensolver residual: {residual}")
    return 1 / radius if radius > 0 else float("inf"), radius, residual


def predictors(degrees: np.ndarray) -> tuple[float, float, float]:
    mean_degree = float(np.mean(degrees))
    second = float(np.mean(degrees**2))
    denom = second - mean_degree
    cebh = mean_degree / denom if denom > 0 else float("nan")
    dimension = 2.0 / mean_degree if mean_degree > 0 else float("nan")
    d3 = float(np.mean(degrees == 3))
    d4 = float(np.mean(degrees >= 4))
    junction = d3 + d4
    anchor = (d3 * HONEYCOMB + d4 * SQUARE) / junction if junction > 0 else float("nan")
    return cebh, dimension, anchor


def connect_components_by_degree_preserving_switches(
    graph: nx.Graph, seed: int
) -> tuple[int, int, float]:
    """Merge components with degree-preserving 2-switches.

    Each merge removes a non-bridge edge from a cyclic anchor component and
    one edge from another component, then reconnects the four endpoints across
    components. The anchor stays connected; both sides of a removed bridge in
    the other component attach to it. Degrees, simplicity, and edge count are
    unchanged.
    """

    rng = random.Random(seed)
    node_components = [set(nodes) for nodes in nx.connected_components(graph)]
    components_before = len(node_components)
    lcc_fraction_before = max(map(len, node_components)) / graph.number_of_nodes()
    if components_before == 1:
        return 1, 0, 1.0

    components: list[dict[str, set[tuple[int, int]] | set[int]]] = []
    for nodes in node_components:
        subgraph = graph.subgraph(nodes)
        root = next(iter(nodes))
        backbone = {tuple(sorted((int(u), int(v)))) for u, v in nx.dfs_edges(subgraph, root)}
        all_edges = {tuple(sorted((int(u), int(v)))) for u, v in subgraph.edges()}
        if not backbone:
            raise RuntimeError("Encountered an isolated node in connected-null repair")
        components.append(
            {
                "nodes": {int(node) for node in nodes},
                "backbone": backbone,
                "surplus": all_edges - backbone,
            }
        )

    anchor_index = max(range(len(components)), key=lambda idx: len(components[idx]["surplus"]))
    anchor = components.pop(anchor_index)
    repairs = 0
    while components:
        anchor_surplus = anchor["surplus"]
        if anchor_surplus:
            target_index = min(
                range(len(components)), key=lambda idx: len(components[idx]["surplus"])
            )
            target = components.pop(target_index)
            donor = rng.choice(tuple(anchor_surplus))
            other = rng.choice(tuple(target["backbone"]))
            anchor_surplus.remove(donor)
            target["backbone"].remove(other)
        else:
            target_index = max(
                range(len(components)), key=lambda idx: len(components[idx]["surplus"])
            )
            target = components.pop(target_index)
            if not target["surplus"]:
                raise RuntimeError(
                    "Insufficient cycle edges for degree-preserving component repair"
                )
            donor = rng.choice(tuple(target["surplus"]))
            other = rng.choice(tuple(anchor["backbone"]))
            target["surplus"].remove(donor)
            anchor["backbone"].remove(other)

        a, b = donor
        c, d = other
        if rng.random() < 0.5:
            c, d = d, c
        graph.remove_edge(a, b)
        graph.remove_edge(c, d)
        graph.add_edge(a, c)
        graph.add_edge(b, d)
        cross_edges = {tuple(sorted((a, c))), tuple(sorted((b, d)))}

        anchor["nodes"].update(target["nodes"])
        anchor["backbone"].update(target["backbone"])
        anchor["backbone"].update(cross_edges)
        anchor["surplus"].update(target["surplus"])
        repairs += 1

    if not nx.is_connected(graph):
        raise RuntimeError("Degree-preserving component repair did not produce a connected graph")
    return components_before, repairs, lcc_fraction_before
