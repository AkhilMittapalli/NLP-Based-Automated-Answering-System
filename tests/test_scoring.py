"""
Unit tests for the scoring module.
"""
import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring import AnswerScorer
from src.preprocessing import TextPreprocessor

class TestAnswerScorer(unittest.TestCase):
    """Test the AnswerScorer class."""
    
    def setUp(self):
        """Set up for tests."""
        self.scorer = AnswerScorer(
            keyword_weight=0.6,
            keyword_order_weight=0.2,
            completeness_weight=0.2,
            partial_match_threshold=0.8
        )
        self.preprocessor = TextPreprocessor()
        
        # Sample reference answer and keywords
        self.ref_answer = "Photosynthesis is the process by which plants convert light energy into chemical energy."
        self.ref_keywords = [
            ("photosynthesis", 1.0),
            ("plants", 0.8),
            ("light energy", 0.9),
            ("chemical energy", 0.9)
        ]
        
        # Sample user answers
        self.excellent_answer = "Photosynthesis is the biological process where plants transform light energy from the sun into chemical energy."
        self.good_answer = "Plants use photosynthesis to convert light energy into chemical energy."
        self.fair_answer = "Photosynthesis is how plants make energy using sunlight."
        self.poor_answer = "Plants make food using sunlight."
        
    def test_perfect_match(self):
        """Test scoring with perfect match."""
        score, breakdown = self.scorer.score_answer(
            self.ref_answer, 
            self.ref_answer, 
            self.ref_keywords
        )
        self.assertEqual(score, 100.0)
        self.assertEqual(breakdown["keyword_score"], 100.0)
        
    def test_excellent_answer(self):
        """Test scoring with excellent but not identical answer."""
        score, breakdown = self.scorer.score_answer(
            self.excellent_answer, 
            self.ref_answer, 
            self.ref_keywords
        )
        self.assertGreaterEqual(score, 90.0)
        
    def test_poor_answer(self):
        """Test scoring with poor answer."""
        score, breakdown = self.scorer.score_answer(
            self.poor_answer, 
            self.ref_answer, 
            self.ref_keywords
        )
        self.assertLess(score, 60.0)
        
    def test_keyword_presence(self):
        """Test the keyword presence scoring component."""
        # Using a method that's not part of the public API
        score, matched = self.scorer._score_keyword_presence(
            self.good_answer,
            dict(self.ref_keywords)
        )
        self.assertGreaterEqual(score, 0.8)
        self.assertEqual(len(matched), 4)  # Should match all keywords
        
    def test_partial_keyword_match(self):
        """Test partial matching of keywords."""
        # Misspell photosynthesis
        test_answer = "Photosyntheris is the process where plants convert light energy to chemical energy."
        score, matched = self.scorer._score_keyword_presence(
            test_answer,
            dict(self.ref_keywords)
        )
        # Should still match "photosynthesis" partially
        self.assertIn("photosynthesis (partial)", matched)
        
    def test_different_weights(self):
        """Test how different weights affect scoring."""
        # Create a scorer that heavily weights keyword presence
        keyword_heavy = AnswerScorer(
            keyword_weight=0.9,
            keyword_order_weight=0.05,
            completeness_weight=0.05
        )
        
        # Create a scorer that heavily weights completeness
        completeness_heavy = AnswerScorer(
            keyword_weight=0.4,
            keyword_order_weight=0.1,
            completeness_weight=0.5
        )
        
        # Score the fair answer with both scorers
        score1, _ = keyword_heavy.score_answer(
            self.fair_answer, 
            self.ref_answer, 
            self.ref_keywords
        )
        
        score2, _ = completeness_heavy.score_answer(
            self.fair_answer, 
            self.ref_answer, 
            self.ref_keywords
        )
        
        # The fair answer has good keywords but is short, so
        # it should score better with keyword_heavy
        self.assertGreater(score1, score2)
        
    def test_preprocessed_input(self):
        """Test scoring with preprocessed tokens."""
        # Preprocess the inputs
        ref_tokens = self.preprocessor.preprocess(self.ref_answer)
        user_tokens = self.preprocessor.preprocess(self.good_answer)
        
        # Score with token lists
        score, breakdown = self.scorer.score_answer(
            user_tokens,
            ref_tokens,
            self.ref_keywords
        )
        
        # Should still get a reasonable score
        self.assertGreaterEqual(score, 70.0)
        
    def test_order_scoring(self):
        """Test the order scoring component."""
        # Create answers with same keywords but different order
        original = "Photosynthesis is how plants convert light energy into chemical energy."
        reordered = "Plants use light energy in photosynthesis to create chemical energy."
        
        # Preprocess
        original_tokens = self.preprocessor.preprocess(original)
        reordered_tokens = self.preprocessor.preprocess(reordered)
        
        # Calculate order score with a fixed set of matched keywords
        matched_keywords = ["photosynthesis", "plants", "light energy", "chemical energy"]
        
        order_score = self.scorer._score_keyword_order(
            original,
            original,
            matched_keywords
        )
        self.assertEqual(order_score, 1.0)  # Perfect order match
        
        order_score = self.scorer._score_keyword_order(
            original,
            reordered,
            matched_keywords
        )
        self.assertLess(order_score, 1.0)  # Should be less than perfect


if __name__ == "__main__":
    unittest.main()