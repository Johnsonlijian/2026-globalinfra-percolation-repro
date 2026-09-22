"""Independent small-graph matrices; fixtures are not empirical evidence."""

import unittest

import networkx as nx
import numpy as np
from canonical_percolation import canonical_curves, endpoint_slope, estimate, peak, sample_counts
from graph_helpers import graph_hash, spectral_threshold
from numpy.testing import assert_allclose


class GraphHelperTests(unittest.TestCase):
    def test_hashimoto_matches_explicit_matrix(self):
        graphs = [
            nx.cycle_graph(3),
            nx.cycle_graph(4),
            nx.complete_graph(4),
            nx.disjoint_union(nx.cycle_graph(3), nx.path_graph(3)),
            nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(4)),
        ]
        leaf = nx.cycle_graph(4)
        leaf.add_edges_from([(0, 4), (4, 5)])
        graphs.append(leaf)
        for graph in graphs:
            edges = np.asarray(list(graph.edges()), dtype=np.int64)
            directed = list(map(tuple, edges)) + [(v, u) for u, v in edges]
            matrix = np.array(
                [[int(v == x and y != u) for x, y in directed] for u, v in directed], dtype=float
            )
            radius = max(abs(np.linalg.eigvals(matrix)))
            pc, measured, residual = spectral_threshold(len(graph), edges)
            assert_allclose(measured, radius, atol=1e-7)
            assert_allclose(pc, 1 / radius, atol=1e-7)
            self.assertLess(residual, 1e-6)

    def test_forest_and_hash_invariants(self):
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        self.assertEqual(graph_hash(3, edges), graph_hash(3, edges[::-1, ::-1]))
        self.assertNotEqual(graph_hash(3, edges), graph_hash(4, edges))
        pc, radius, residual = spectral_threshold(3, edges)
        self.assertTrue(np.isinf(pc))
        self.assertEqual((radius, residual), (0.0, 0.0))

    def test_bootstrap_and_hybrid_endpoint_exact_small_fixture(self):
        edges = np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64)
        result, stored = estimate(4, edges, n_orders=64, n_bootstrap=31, seed=731, step=0.025)
        _, derivative = canonical_curves(sample_counts(4, edges, 64, 731), stored["grid"])
        exact, _ = endpoint_slope(4, edges)
        derivative[-1] = exact
        rng = np.random.default_rng(731 + 7919)
        weights = rng.multinomial(64, np.full(64, 1 / 64), 31)
        draws = [peak(stored["grid"], column) for column in (derivative @ (weights.T / 64)).T]
        assert_allclose(draws, stored["bootstrap_pc"], atol=1e-12)
        assert_allclose([result["pc_low"], result["pc_high"]], np.quantile(draws, [0.025, 0.975]))
        assert_allclose(stored["derivative"][-1], exact)
