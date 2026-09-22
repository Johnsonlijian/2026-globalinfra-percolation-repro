"""Regression tests for the final provenance and numerical diagnostic repairs."""
import unittest
import os
import tempfile

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from common import PACKAGE
from screening import evaluate_allowance


class RevisionDiagnosticTests(unittest.TestCase):
    def test_component_bookkeeping_against_exact_networkx_sizes(self):
        os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "percolation-test-numba"))
        from response_definition_verified import self_test
        self_test()

    def test_retrospective_split_keeps_all_cities_and_frozen_allowance(self):
        ext = pd.read_csv(PACKAGE / 'data/external.csv')
        ref = pd.read_csv(PACKAGE / 'data/observed.csv').query("domain == 'road'")
        q = np.sort(abs(ref.pc - ref.pc_dimension))[68]
        prior = ext.instance.isin(['Stockholm', 'Cape Town'])
        self.assertEqual((int((~prior).sum()), int(prior.sum())), (19, 2))
        summaries = pd.read_csv(PACKAGE / 'data/external_cohort_split_summary.csv').set_index('cohort')
        for key, subset in [('external_primary19', ext[~prior]),
                            ('external_sensitivity2', ext[prior]),
                            ('external_acquired21', ext)]:
            saved = summaries.loc[key]
            self.assertEqual(len(subset), saved.n)
            self.assertAlmostEqual(saved.q, q, places=14)
            result = evaluate_allowance(1-subset.pc_dimension, 1-subset.pc, q)
            self.assertAlmostEqual(result.retained.mean(), saved.retained, places=14)
            self.assertEqual(saved.covered, (abs(subset.pc-subset.pc_dimension) <= q).sum())

    def test_control_ladder_is_complete_and_uses_actual_control_response(self):
        frame = pd.read_csv(PACKAGE / 'data/control_ladders.csv')
        self.assertEqual(len(frame), 272)
        assert_allclose(frame.null_gap, frame.pc_control-frame.pc_cebh, atol=1e-14)
        assert_allclose(frame.organization_residual, frame.pc_observed-frame.pc_control, atol=1e-14)
        ladder = frame[frame.ladder.eq('order')]
        self.assertEqual(ladder.instance.nunique(), 24)
        self.assertEqual(ladder.groupby(['instance', 'replicate']).size().unique().tolist(), [5])
        self.assertEqual(sorted(ladder.orders.unique()), [256, 512, 1024, 2048, 4096])
        self.assertTrue(frame.degree_sequence_exact.all())
        self.assertTrue(frame.organization_residual.gt(0).all())
        self.assertTrue(ladder.groupby(['instance', 'replicate']).graph_sha256.nunique().eq(1).all())

    def test_alternative_response_values_and_gap_identity(self):
        frame = pd.read_csv(PACKAGE / 'data/response_definition_check.csv')
        self.assertEqual(len(frame), 91)
        for col in ['p_star_s1', 'p_peak_s2', 'p_peak_susceptibility',
                    'p_star_order_mean', 'p_star_order_median']:
            assert_allclose(frame['gap_deg_'+col], frame[col]-frame.h_deg, atol=1e-14)
            self.assertTrue(frame[col].between(0, 1).all())
            self.assertTrue(frame['gap_deg_'+col].gt(0).all())
        self.assertTrue(frame.orders.eq(512).all())

    def test_scigrid_numerator_and_denominator_refer_to_simple_edges(self):
        frame = pd.read_csv(PACKAGE / 'data/power_scigrid_de.csv')
        assert_allclose(frame.chi_chord, frame.crossing_pairs_chord/frame.n_edges, atol=1e-14)
        self.assertTrue(frame.chords_used.eq(frame.n_edges).all())
        full = frame.set_index('scope').loc['germany_full']
        self.assertEqual((full.n_nodes, full.n_edges, full.crossing_pairs_chord), (585, 801, 209))
        self.assertLess(full.gap_2d, -0.061536786614714734)

    def test_shifted_predictor_own_allowance_is_not_the_main_allowance(self):
        shared = pd.read_csv(PACKAGE/'data/external_shift_comparator.csv').set_index('cohort')
        own = pd.read_csv(PACKAGE/'data/external_shift_equal_rank.csv').set_index('cohort')
        self.assertEqual(shared.loc['external_primary', 'covered'], 18)
        self.assertEqual(own.loc['external_primary', 'covered'], 17)
        self.assertLess(own.loc['external_primary', 'q'], shared.loc['external_primary', 'q'])
        self.assertGreater(own.loc['external_primary', 'conditional_unsafe'], 0)
