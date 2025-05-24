"""
Semantic analysis module for more advanced answer evaluation.
"""
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticAnalyzer:
    """Analyze semantic similarity between texts using spaCy."""
    
    def __init__(self, model_name="en_core_web_md"):
        """
        Initialize the semantic analyzer.
        
        Args:
            model_name: spaCy model to use (default: en_core_web_md)
        """
        try:
            self.nlp = spacy.load(model_name)
            print(f"Loaded spaCy model: {model_name}")
        except OSError:
            print(f"Downloading spaCy model: {model_name}")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
    
    def get_sentence_embeddings(self, text):
        """
        Get sentence-level embeddings for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of sentence embedding vectors
        """
        doc = self.nlp(text)
        return [sent.vector for sent in doc.sents]
    
    def get_document_embedding(self, text):
        """
        Get document-level embedding for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Document embedding vector
        """
        doc = self.nlp(text)
        return doc.vector
    
    def compute_similarity_with_threshold(self, text1, text2, method="document", threshold=0.3):
        """
        Compute semantic similarity with threshold filtering.
        
        Args:
            text1: First text
            text2: Second text
            method: Similarity method ('document', 'sentence', or 'token')
            threshold: Minimum similarity score to consider (0-1)
            
        Returns:
            Similarity score between 0 and 1, or 0 if below threshold
        """
        similarity = self.compute_similarity(text1, text2, method)
        
        # Apply threshold
        if similarity < threshold:
            return 0.0
        
        return similarity
    
    def compute_similarity(self, text1, text2, method="document"):
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            method: Similarity method ('document', 'sentence', or 'token')
            
        Returns:
            Similarity score between 0 and 1
        """
        if method == "document":
            # Document-level similarity
            vec1 = self.get_document_embedding(text1)
            vec2 = self.get_document_embedding(text2)
            
            # Handle zero vectors
            if np.all(vec1 == 0) or np.all(vec2 == 0):
                return 0.0
                
            return cosine_similarity([vec1], [vec2])[0][0]
            
        elif method == "sentence":
            # Sentence-level similarity (average of best matches)
            sent_vecs1 = self.get_sentence_embeddings(text1)
            sent_vecs2 = self.get_sentence_embeddings(text2)
            
            if not sent_vecs1 or not sent_vecs2:
                return 0.0
                
            # For each sentence in text1, find the best matching sentence in text2
            similarities = []
            for vec1 in sent_vecs1:
                if np.all(vec1 == 0):
                    continue
                    
                best_sim = 0
                for vec2 in sent_vecs2:
                    if np.all(vec2 == 0):
                        continue
                        
                    sim = cosine_similarity([vec1], [vec2])[0][0]
                    if sim > best_sim:
                        best_sim = sim
                        
                if best_sim > 0:
                    similarities.append(best_sim)
            
            # Return average of best similarities
            return np.mean(similarities) if similarities else 0.0
            
        elif method == "token":
            # Token-level similarity using spaCy's built-in similarity
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            
            return doc1.similarity(doc2)
            
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def analyze_completeness(self, user_answer, reference_answer, method="sentence"):
        """
        Analyze how complete the user's answer is compared to the reference.
        
        Args:
            user_answer: User's answer text
            reference_answer: Reference answer text
            method: Similarity method to use
            
        Returns:
            Completeness score between 0 and 1
        """
        # Get similarity score
        similarity = self.compute_similarity(user_answer, reference_answer, method)
        
        # Additional analysis could be added here, such as:
        # - Coverage of key concepts
        # - Depth vs. breadth of answer
        # - Structure and coherence
        
        return similarity
    
    def extract_key_concepts(self, text):
        """
        Extract key concepts from text using NER and noun chunks.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of key concepts
        """
        doc = self.nlp(text)
        
        # Extract named entities
        entities = [ent.text for ent in doc.ents]
        
        # Extract noun chunks (normalized)
        chunks = [chunk.text for chunk in doc.noun_chunks]
        
        # Extract important noun tokens based on POS and dependency
        important_nouns = [token.text for token in doc 
                          if token.pos_ in ("NOUN", "PROPN") 
                          and token.dep_ in ("nsubj", "dobj", "pobj", "attr")]
        
        # Combine and remove duplicates
        all_concepts = entities + chunks + important_nouns
        unique_concepts = list(set(all_concepts))
        
        return unique_concepts

    def analyze_concept_coverage(self, user_answer, reference_answer, threshold=0.5):
        """
        Analyze how well the user's answer covers key concepts from reference.
        
        Args:
            user_answer: User's answer text
            reference_answer: Reference answer text
            threshold: Minimum coverage score to consider (0-1)
            method: Similarity method to use
        Returns:
            Concept coverage score between 0 and 1 and list of matched concepts
        """
        # Extract key concepts
        user_concepts = set(self.extract_key_concepts(user_answer.lower()))
        ref_concepts = set(self.extract_key_concepts(reference_answer.lower()))
        
        if not ref_concepts:
            return 0.0, []
            
        # Find matches (including partial matches)
        matched_concepts = []
        for ref_concept in ref_concepts:
            # Check for exact match
            if ref_concept.lower() in user_concepts:
                matched_concepts.append(ref_concept)
                continue
                
            # Check for partial matches
            for user_concept in user_concepts:
                if (ref_concept.lower() in user_concept.lower() or 
                    user_concept.lower() in ref_concept.lower()):
                    matched_concepts.append(ref_concept)
                    break
        
        # Calculate coverage score
        coverage_score = len(matched_concepts) / len(ref_concepts)
        
        return coverage_score, matched_concepts