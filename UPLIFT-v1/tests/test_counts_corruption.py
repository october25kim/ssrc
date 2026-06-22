import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift.priors.counts import counts_to_prior, l1_distance, make_balanced_counts
from uplift.priors.corruption import corrupt_counts, corrupt_open_world_prior, corrupt_prior
from uplift.priors.recovery import mean_prior_error, recover_known_priors


class CountsCorruptionTest(unittest.TestCase):
    def test_balanced_counts_preserve_total(self):
        self.assertEqual(sum(make_balanced_counts(3, 10)), 10)

    def test_counts_to_prior_normalizes(self):
        self.assertEqual(counts_to_prior([1, 1, 2]), [0.25, 0.25, 0.5])

    def test_mix_uniform_prior_preserves_simplex(self):
        prior = corrupt_prior([0.9, 0.1], strength=0.5)
        self.assertAlmostEqual(sum(prior), 1.0)
        self.assertLess(prior[0], 0.9)
        self.assertGreater(prior[1], 0.1)

    def test_corrupt_counts_preserves_total(self):
        counts = [90, 10]
        self.assertEqual(sum(corrupt_counts(counts, strength=0.5)), sum(counts))

    def test_open_world_corruption_matches_omitted_unknown_bias(self):
        clean = [0.8, 0.2]
        routing = [0.1, 0.9]
        gamma = 0.25
        observed = corrupt_open_world_prior(clean, gamma=gamma, routing_prior=routing)
        self.assertAlmostEqual(l1_distance(observed, clean), gamma * l1_distance(routing, clean))

    def test_recovery_improves_simple_controlled_case(self):
        clean = [[0.8, 0.2], [0.2, 0.8]]
        routing = [0.5, 0.5]
        gammas = [0.2, 0.2]
        observed = [corrupt_open_world_prior(row, gamma=g, routing_prior=routing) for row, g in zip(clean, gammas)]
        recovered = recover_known_priors(observed, gammas=gammas, routing_prior=routing).recovered_priors
        self.assertLess(mean_prior_error(recovered, clean), mean_prior_error(observed, clean))


if __name__ == "__main__":
    unittest.main()
