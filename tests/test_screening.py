import unittest

import numpy as np
from numpy.testing import assert_allclose
from screening import allowance, evaluate_allowance, integrated_errors


class ScreeningTests(unittest.TestCase):
    def test_integrated_against_dense_grid(self):
        predicted = np.array([0.1, 0.2, 0.4, 0.8, 0.4])
        observed = np.array([0.4, 0.1, 0.5, 0.6, 0.4])
        targets = np.linspace(0.15, 0.55, 200001)
        unsafe, safe = integrated_errors(predicted, observed)
        assert_allclose(
            unsafe,
            ((predicted[:, None] > targets) & (observed[:, None] < targets)).mean(axis=1),
            atol=1e-5,
        )
        assert_allclose(
            safe,
            ((predicted[:, None] < targets) & (observed[:, None] > targets)).mean(axis=1),
            atol=1e-5,
        )
        evaluated = evaluate_allowance(predicted, observed, 0.03)
        escalated = abs(predicted[:, None] - targets) <= 0.03
        direct_u = (
            ((predicted[:, None] > targets) & (observed[:, None] < targets)) & ~escalated
        ).mean(axis=1)
        assert_allclose(evaluated.unsafe_retained, direct_u, atol=1e-5)
        assert_allclose(evaluated.escalated, escalated.mean(axis=1), atol=1e-5)

    def test_finite_rank_allowance(self):
        self.assertEqual(allowance(np.arange(71)), 68)
        self.assertTrue(np.isinf(allowance(np.arange(4))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
