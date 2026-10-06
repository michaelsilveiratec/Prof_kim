from __future__ import annotations

import unittest

from src.bayes import BayesianWordPredictor
from src.config import UNK_TOKEN


class BayesianWordPredictorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = BayesianWordPredictor(alpha=1.0)
        self.predictor.fit(
            [
                ["o", "aluno", "estudou", "estatística"],
                ["o", "aluno", "estudou", "estatística"],
                ["o", "aluno", "estudou", "estatística"],
                ["a", "aluna", "leu", "probabilidade"],
            ]
        )

    def test_posterior_sums_to_one(self) -> None:
        distribution = self.predictor.next_word_distribution(["o", "aluno", "estudou"])
        self.assertAlmostEqual(sum(distribution.values()), 1.0, places=8)

    def test_laplace_smoothing_keeps_unknown_continuation_possible(self) -> None:
        distribution = self.predictor.next_word_distribution(["o", "aluno"])
        self.assertGreater(distribution[UNK_TOKEN], 0.0)
        self.assertAlmostEqual(sum(distribution.values()), 1.0, places=8)

    def test_prefers_frequent_continuation(self) -> None:
        prediction = self.predictor.predict_word("o aluno estudou")
        self.assertEqual(prediction.word, "estatística")
        self.assertGreater(prediction.probability, 0.0)

    def test_unseen_context_does_not_collapse(self) -> None:
        distribution = self.predictor.next_word_distribution(
            ["contexto", "inédito", "qualquer"]
        )
        prediction = self.predictor.predict_word("contexto inédito qualquer")
        self.assertIsNotNone(prediction.word)
        self.assertGreater(prediction.probability, 0.0)
        self.assertTrue(all(probability > 0 for probability in distribution.values()))
        self.assertAlmostEqual(sum(distribution.values()), 1.0, places=8)

    def test_empty_phrase_uses_prior(self) -> None:
        prediction = self.predictor.predict_word("")
        counts = self.predictor.model.unigram_counts
        self.assertEqual(counts[prediction.word], max(counts.values()))
        self.assertAlmostEqual(prediction.confidence_pct, prediction.probability * 100)

    def test_rejects_non_positive_alpha(self) -> None:
        with self.assertRaises(ValueError):
            BayesianWordPredictor(alpha=0)

    def test_rejects_weights_that_do_not_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            BayesianWordPredictor(higher_order_weight=0.5, lower_order_weight=0.3)


if __name__ == "__main__":
    unittest.main()
