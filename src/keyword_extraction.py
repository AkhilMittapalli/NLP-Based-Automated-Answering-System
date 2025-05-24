"""
Keyword extraction functionality for answer evaluation.
"""
import nltk
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

class KeywordExtractor:
    """Extract keywords from text using various methods."""
    
    def __init__(self, method="frequency", max_keywords=10):
        """
        Initialize the keyword extractor.
        
        Args:
            method: Extraction method ('frequency', 'tfidf', or 'ngram')
            max_keywords: Maximum number of keywords to extract
        """
        self.method = method
        self.max_keywords = max_keywords
    
    def extract_keywords(self, text, reference_texts=None):
        """
        Extract keywords from the given text.
        
        Args:
            text: Text to extract keywords from
            reference_texts: Reference texts for TF-IDF comparison (required for tfidf method)
            
        Returns:
            List of keywords
        """
        if not text:
            return []
            
        if self.method == "frequency":
            return self._extract_by_frequency(text)
        elif self.method == "tfidf":
            if reference_texts is None:
                raise ValueError("Reference texts required for TF-IDF extraction")
            return self._extract_by_tfidf(text, reference_texts)
        elif self.method == "ngram":
            return self._extract_ngrams(text)
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")
    
    def _extract_by_frequency(self, tokens):
        """Extract keywords based on frequency."""
        if isinstance(tokens, str):
            # If input is a string, tokenize it
            tokens = nltk.word_tokenize(tokens.lower())
            
        # Count token frequencies
        word_counts = Counter(tokens)
        
        # Return the most common tokens
        return [word for word, count in word_counts.most_common(self.max_keywords)]
    
    def _extract_by_tfidf(self, text, reference_texts):
        """Extract keywords using TF-IDF."""
        # Create a corpus with the text and reference texts
        corpus = [text] + reference_texts
        
        # Calculate TF-IDF
        vectorizer = TfidfVectorizer(max_features=self.max_keywords * 2)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get the TF-IDF scores for the target text (first in corpus)
        tfidf_scores = list(zip(feature_names, tfidf_matrix[0].toarray()[0]))
        
        # Sort by score and return top keywords
        sorted_keywords = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_keywords[:self.max_keywords]]
    
    def _extract_ngrams(self, tokens, n=2):
        """Extract n-grams as keywords."""
        if isinstance(tokens, str):
            # If input is a string, tokenize it
            tokens = nltk.word_tokenize(tokens.lower())
            
        # Generate n-grams
        n_grams = list(ngrams(tokens, n))
        
        # Count n-gram frequencies
        ngram_counts = Counter(n_grams)
        
        # Return the most common n-grams as joined strings
        return [' '.join(gram) for gram, count in ngram_counts.most_common(self.max_keywords)]
    
    def get_keyword_importance(self, keywords, reference_keywords):
        """
        Calculate importance scores for keywords based on reference keywords.
        
        Args:
            keywords: List of extracted keywords
            reference_keywords: List of reference keywords with importance
            
        Returns:
            Dictionary mapping keywords to importance scores
        """
        # If reference_keywords is a list without scores, assign equal weights
        if all(isinstance(k, str) for k in reference_keywords):
            ref_dict = {k: 1.0 for k in reference_keywords}
        else:
            # Assume reference_keywords is a list of (keyword, score) tuples
            ref_dict = dict(reference_keywords)
            
        # Calculate importance for each keyword
        importance = {}
        for kw in keywords:
            if kw in ref_dict:
                importance[kw] = ref_dict[kw]
            else:
                # Check for partial matches
                partial_matches = [r for r in ref_dict if kw in r or r in kw]
                if partial_matches:
                    # Use the maximum score of partial matches
                    importance[kw] = max(ref_dict[r] for r in partial_matches)
                else:
                    importance[kw] = 0.0
                    
        return importance