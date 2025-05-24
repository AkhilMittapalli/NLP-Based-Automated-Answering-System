"""
Scoring module for answer evaluation system.
"""
from collections import Counter
import numpy as np
from difflib import SequenceMatcher

class AnswerScorer:
    """Score user answers based on keyword presence and other metrics."""
    
    def __init__(self, 
                 keyword_weight=0.3,
                 keyword_order_weight=0.1, 
                 completeness_weight=0.2,
                 semantic_weight=0.4,
                 partial_match_threshold=0.8,
                 semantic_analyzer=None):
        """
        Initialize the answer scorer with specified weights.
        
        Args:
            keyword_weight: Weight for keyword matching
            keyword_order_weight: Weight for keyword order matching
            completeness_weight: Weight for answer completeness
            semantic_weight: Weight for semantic similarity
            partial_match_threshold: Threshold for partial keyword matching
            semantic_analyzer: SemanticAnalyzer instance (optional)
        """
        self.keyword_weight = keyword_weight
        self.keyword_order_weight = keyword_order_weight
        self.completeness_weight = completeness_weight
        self.semantic_weight = semantic_weight
        self.partial_match_threshold = partial_match_threshold
        self.semantic_analyzer = semantic_analyzer
        
        # Validate that weights sum to 1.0
        total_weight = keyword_weight + keyword_order_weight + completeness_weight + semantic_weight
        assert abs(total_weight - 1.0) < 0.001, f"Weights must sum to 1.0, got {total_weight}"
    
    def score_answer(self, user_answer, reference_answer, reference_keywords):
        """
        Score a user answer against a reference answer.
        
        Args:
            user_answer: User's answer text (or tokens)
            reference_answer: Reference answer text (or tokens)
            reference_keywords: List of keywords from reference answer with importance
            
        Returns:
            Score (0-100), breakdown dictionary
        """
        # Convert to string if tokens provided
        user_text = " ".join(user_answer) if isinstance(user_answer, list) else user_answer
        ref_text = " ".join(reference_answer) if isinstance(reference_answer, list) else reference_answer
        
        # Get reference keywords as dict if provided as list of tuples
        if reference_keywords and isinstance(reference_keywords[0], tuple) or (isinstance(reference_keywords[0], list) and len(reference_keywords[0]) == 2):
            ref_keywords = {k: v for k, v in reference_keywords}
        else:  # If just a list of strings
            ref_keywords = {}
            for item in reference_keywords:
                if isinstance(item, list) and len(item) == 2:
                    ref_keywords[item[0]] = item[1]
                else:
                    ref_keywords[item] = 1.0
        
        # Score keyword presence
        keyword_score, matched_keywords = self._score_keyword_presence(
            user_text, ref_keywords)
        
        # Score keyword order
        order_score = self._score_keyword_order(user_text, ref_text, matched_keywords)
        
        # Score traditional completeness (based on length)
        trad_completeness_score = self._score_completeness(user_text, ref_text)
        
        # Score semantic similarity (if semantic analyzer is available)
        semantic_score = 0.0
        concept_coverage_score = 0.0
        matched_concepts = []
        
        if self.semantic_analyzer is not None:
            # Document-level similarity
            semantic_score = self.semantic_analyzer.compute_similarity(
                user_text, ref_text, method="document")
            
            # Concept coverage
            concept_coverage_score, matched_concepts = self.semantic_analyzer.analyze_concept_coverage(
                user_text, ref_text)
            
            # Use semantic analysis for completeness score
            completeness_score = (trad_completeness_score + semantic_score + concept_coverage_score) / 3
        else:
            # If no semantic analyzer, fall back to traditional completeness
            completeness_score = trad_completeness_score
            
        # Calculate final score
        final_score = (
            self.keyword_weight * keyword_score +
            self.keyword_order_weight * order_score +
            self.completeness_weight * completeness_score +
            self.semantic_weight * semantic_score
        ) * 100
        
        # Create score breakdown
        breakdown = {
            "keyword_score": keyword_score * 100,
            "order_score": order_score * 100,
            "completeness_score": completeness_score * 100,
            "semantic_score": semantic_score * 100,
            "concept_coverage_score": concept_coverage_score * 100,
            "matched_keywords": matched_keywords,
            "matched_concepts": matched_concepts,
            "total_score": final_score
        }
        
        return final_score, breakdown
    
    def _score_keyword_presence(self, user_text, reference_keywords):
        """Score the presence of keywords in the user's answer."""
        total_importance = sum(reference_keywords.values())
        matched_importance = 0
        matched_keywords = []
        
        for keyword, importance in reference_keywords.items():
            if keyword.lower() in user_text.lower():
                matched_importance += importance
                matched_keywords.append(keyword)
            else:
                # Check for partial matches
                for word in user_text.lower().split():
                    similarity = SequenceMatcher(None, word, keyword.lower()).ratio()
                    if similarity >= self.partial_match_threshold:
                        # Add partial credit based on similarity
                        matched_importance += importance * similarity
                        matched_keywords.append(f"{keyword} (partial)")
                        break
        
        if total_importance == 0:
            return 0, matched_keywords
        
        return matched_importance / total_importance, matched_keywords
    
    def _score_keyword_order(self, user_text, reference_text, matched_keywords):
        """Score the order of keywords in the user's answer compared to reference."""
        if not matched_keywords:
            return 0
            
        # Find positions of matched keywords in both texts
        user_positions = {}
        ref_positions = {}
        
        for keyword in matched_keywords:
            # Remove "(partial)" suffix if present
            clean_keyword = keyword.split(" (partial)")[0]
            
            # Find position in user text
            pos = user_text.lower().find(clean_keyword.lower())
            if pos >= 0:
                user_positions[clean_keyword] = pos
                
            # Find position in reference text
            pos = reference_text.lower().find(clean_keyword.lower())
            if pos >= 0:
                ref_positions[clean_keyword] = pos
        
        # Calculate order correlation
        common_keywords = set(user_positions.keys()) & set(ref_positions.keys())
        if len(common_keywords) < 2:
            return 1.0  # Not enough keywords to compare order
            
        # Convert positions to ranks
        user_ranks = {k: i for i, k in enumerate(sorted(common_keywords, key=lambda k: user_positions[k]))}
        ref_ranks = {k: i for i, k in enumerate(sorted(common_keywords, key=lambda k: ref_positions[k]))}
        
        # Calculate rank correlation
        n = len(common_keywords)
        sum_sq_diff = sum((user_ranks[k] - ref_ranks[k]) ** 2 for k in common_keywords)
        
        # Spearman's rank correlation formula
        correlation = 1 - (6 * sum_sq_diff) / (n * (n**2 - 1))
        
        # Convert to 0-1 scale (correlation is in [-1, 1])
        return (correlation + 1) / 2
    
    def _score_completeness(self, user_text, reference_text):
        """
        Score completeness based on length ratio.
        This is a simple baseline that will be enhanced by semantic analysis.
        """
        user_words = len(user_text.split())
        ref_words = len(reference_text.split())
        
        if ref_words == 0:
            return 0.0
            
        # Cap at 1.0 to avoid rewarding excessive verbosity
        return min(1.0, user_words / ref_words)