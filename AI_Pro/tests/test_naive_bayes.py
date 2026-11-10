"""Unit tests for Naive Bayes classifier and word influence attribution."""

import unittest
import math
import os
import tempfile
from src.preprocessor import TextPreprocessor
from src.naive_bayes import NaiveBayesClassifier
from src.evaluate import calculate_metrics, benchmark_against_sklearn


class TestNaiveBayes(unittest.TestCase):
    """Test cases for Naive Bayes implementation and explainability logic."""

    def setUp(self):
        self.preprocessor = TextPreprocessor(lowercase=True)
        self.train_texts = [
            "Win free cash prize right now claim award",
            "Urgent claim free lottery money reward bonus",
            "Exclusive free cash coupon win today click here",
            "Hi team meeting tomorrow morning in conference room",
            "Please review the attached project document and slides",
            "Can you send over the engineering report by Friday thanks"
        ]
        self.train_labels = [
            "spam",
            "spam",
            "spam",
            "ham",
            "ham",
            "ham"
        ]
        self.classifier = NaiveBayesClassifier(alpha=1.0, preprocessor=self.preprocessor)
        self.classifier.fit(self.train_texts, self.train_labels)

    def test_vocabulary_built(self):
        """Ensure vocabulary size is positive and contains expected words."""
        self.assertGreater(self.classifier.vocab_size, 10)
        self.assertIn("free", self.classifier.vocab)
        self.assertIn("meeting", self.classifier.vocab)

    def test_spam_prediction(self):
        """Test that blatant spam text is classified as spam."""
        test_email = "Claim your free cash prize award right now"
        pred = self.classifier.predict(test_email)
        self.assertEqual(pred, "spam")
        probs = self.classifier.predict_proba_single(test_email)
        self.assertGreater(probs["spam"], 0.70)

    def test_ham_prediction(self):
        """Test that normal workplace communication is classified as ham."""
        test_email = "Hi team, please review the project slides for our meeting tomorrow"
        pred = self.classifier.predict(test_email)
        self.assertEqual(pred, "ham")
        probs = self.classifier.predict_proba_single(test_email)
        self.assertGreater(probs["ham"], 0.70)

    def test_word_influence_exact_additivity(self):
        """Verify that prior log-odds + sum of word influences equals total log-odds exactly."""
        test_email = "Free cash prize for tomorrow team meeting"
        explanation = self.classifier.explain(test_email)

        prior = explanation["prior_log_odds"]
        net_words = explanation["net_word_evidence"]
        total_log_odds = explanation["total_log_odds"]

        # Mathematical property: delta = prior + sum(phi_i)
        self.assertAlmostEqual(prior + net_words, total_log_odds, places=5)

        # Verify probability derived from total log-odds matches sigmoid
        expected_p_spam = 1.0 / (1.0 + math.exp(-total_log_odds))
        self.assertAlmostEqual(explanation["probabilities"]["spam"], expected_p_spam, places=5)

    def test_most_influential_words_ranking(self):
        """Verify that words with highest positive LLR are ranked top for spam."""
        test_email = "Win free money today"
        explanation = self.classifier.explain(test_email)
        self.assertEqual(explanation["predicted_class"], "spam")

        top_words = [item["word"] for item in explanation["top_influential_words"]]
        self.assertTrue(any(w in ["free", "win", "prize", "money"] for w in top_words[:3]))

    def test_counter_evidence_detection(self):
        """Verify that ham words in a spam email are correctly reported as counter-evidence."""
        test_email = "Claim free cash prize now, also attend project meeting"
        explanation = self.classifier.explain(test_email)

        opposing_words = [item["word"] for item in explanation["top_opposing_words"]]
        # 'meeting' or 'project' should be opposing evidence (ham-leaning)
        self.assertTrue("meeting" in opposing_words or "project" in opposing_words)

    def test_oov_handling(self):
        """Verify that unseen out-of-vocabulary words do not crash and have neutral impact."""
        test_email = "Zyxwvutsrqponmlkjihgfedcba completely unknown token"
        explanation = self.classifier.explain(test_email)
        self.assertIn("zyxwvutsrqponmlkjihgfedcba", explanation["oov_words"])

    def test_save_and_load(self):
        """Verify model serialization and deserialization retains identical parameters."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.classifier.save(tmp_path)
            loaded_model = NaiveBayesClassifier.load(tmp_path, preprocessor=self.preprocessor)

            test_email = "Win free prize money"
            p1 = self.classifier.predict_proba_single(test_email)
            p2 = loaded_model.predict_proba_single(test_email)

            self.assertAlmostEqual(p1["spam"], p2["spam"], places=6)
            self.assertAlmostEqual(p1["ham"], p2["ham"], places=6)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_metrics_calculation(self):
        """Verify accuracy, precision, recall, and F1 calculations."""
        y_true = ["spam", "spam", "ham", "ham"]
        y_pred = ["spam", "ham", "ham", "ham"]
        # TP=1, FN=1, TN=2, FP=0
        # Accuracy: 3/4 = 0.75
        # Precision: 1/(1+0) = 1.0
        # Recall: 1/(1+1) = 0.5
        # F1: 2*(1*0.5)/(1+0.5) = 1/1.5 = 0.6667
        metrics = calculate_metrics(y_true, y_pred)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1_score"], 2.0 / 3.0, places=4)


if __name__ == "__main__":
    unittest.main()
