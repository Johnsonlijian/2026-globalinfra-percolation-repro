"""Independent enumerations and algebraic checks for the canonical estimator."""

import math
import unittest
from itertools import combinations, permutations

import networkx as nx
import numpy as np
from canonical_percolation import (
    canonical_curves,
    count_curve,
    endpoint_slope,
    sample_counts,
    strict_flips,
)
from numpy.testing import assert_allclose


def enumerated_count_means(n, edges):
    values = []
    for k in range(len(edges) + 1):
        observations = []
        for selected in combinations(range(len(edges)), k):
            g = nx.Graph()
            g.add_nodes_from(range(n))
            g.add_edges_from(edges[list(selected)].reshape(-1, 2))
            observations.append(max(map(len, nx.connected_components(g))) / n)
        values.append(np.mean(observations))
    return np.array(values)


class CanonicalTests(unittest.TestCase):
    def test_all_orders_equal_all_subsets(self):
        for n, edges in [
            (3, [(0, 1), (1, 2), (0, 2)]),
            (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
            (5, [(0, 1), (2, 3), (3, 4)]),
        ]:
            edges = np.array(edges, dtype=np.int64)
            curves = np.array(
                [
                    count_curve(n, edges, np.array(order))
                    for order in permutations(range(len(edges)))
                ]
            )
            exact = enumerated_count_means(n, edges)
            assert_allclose(curves.mean(axis=0), exact, atol=1e-14)
            grid = np.linspace(0, 1, 101)
            curve, derivative = canonical_curves(exact[None, :], grid)
            m = len(edges)
            direct = np.array(
                [
                    sum(
                        math.comb(m, k) * p**k * (1 - p) ** (m - k) * exact[k] for k in range(m + 1)
                    )
                    for p in grid
                ]
            )
            assert_allclose(curve[:, 0], direct, atol=1e-13)
            fine_grid = np.linspace(0.001, 0.999, 999)
            fine_curve, fine_d = canonical_curves(exact[None, :], fine_grid)
            assert_allclose(
                np.gradient(fine_curve[:, 0], fine_grid)[1:-1], fine_d[1:-1, 0], atol=1e-5
            )

    def test_single_edge_and_disconnected_endpoint(self):
        counts = sample_counts(4, np.array([[0, 1]], dtype=np.int64), 3, 9)
        grid = np.linspace(0, 1, 11)
        curve, derivative = canonical_curves(counts, grid)
        assert_allclose(curve[:, 0], (1 + grid) / 4)
        assert_allclose(derivative, 0.25)

    def test_nested_order_reproducibility(self):
        edges = np.array([(0, 1), (1, 2), (0, 2)], dtype=np.int64)
        assert_allclose(sample_counts(3, edges, 4, 55), sample_counts(3, edges, 8, 55)[:4])

    def test_exact_tie_is_not_a_flip(self):
        unsafe, safe = strict_flips(np.array([0.50]), np.array([1 - 0.675]), np.array([0.325]))
        self.assertFalse(unsafe.any() or safe.any())
        unsafe, safe = strict_flips(
            np.array([0.50, 0.20]), np.array([0.30, 0.40]), np.array([0.35])
        )
        assert_allclose(unsafe[:, 0], [True, False])
        assert_allclose(safe[:, 0], [False, True])

    def test_exact_endpoint_by_deleting_every_edge(self):
        for seed in range(30):
            graph = nx.gnp_random_graph(15, 0.15 + 0.01 * (seed % 10), seed=seed)
            if graph.number_of_edges() == 0:
                continue
            original = max(map(len, nx.connected_components(graph))) / len(graph)
            direct = 0.0
            for u, v in list(graph.edges()):
                graph.remove_edge(u, v)
                direct += original - max(map(len, nx.connected_components(graph))) / len(graph)
                graph.add_edge(u, v)
            measured, _ = endpoint_slope(len(graph), np.array(graph.edges(), dtype=np.int64))
            self.assertAlmostEqual(measured, direct, places=13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
