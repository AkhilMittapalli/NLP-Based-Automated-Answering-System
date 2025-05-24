"""
Evaluation system for free-text answers.
"""
import json
from preprocessing import TextPreprocessor
from keyword_extraction import KeywordExtractor
from scoring import AnswerScorer

class AnswerEvaluator:
    """Main class for evaluating free-text answers."""
    
    def __init__(self, 
                 preprocessor=None,
                 keyword_extractor=None,
                 scorer=None,
                 reference_data=None):
        """
        Initialize the answer evaluator system.
        
        Args:
            preprocessor: Text preprocessor instance
            keyword_extractor: Keyword extractor instance
            scorer: Answer scorer instance
            reference_data: Path to reference data or dict of Q&A pairs
        """
        # Initialize components with defaults if not provided
        self.preprocessor = preprocessor or TextPreprocessor()
        self.keyword_extractor = keyword_extractor or KeywordExtractor(method="frequency")
        self.scorer = scorer or AnswerScorer()
        
        # Load reference data if provided
        self.reference_data = {}
        if reference_data:
            self.load_reference_data(reference_data)
    
    def load_reference_data(self, reference_data):
        """
        Load reference questions and answers.
        
        Args:
            reference_data: Path to JSON file or dictionary with Q&A pairs
        """
        if isinstance(reference_data, str):
            # Load from file
            with open(reference_data, 'r') as f:
                self.reference_data = json.load(f)
        else:
            # Assume it's a dictionary
            self.reference_data = reference_data
            
        # Extract keywords for all reference answers
        for question_id, data in self.reference_data.items():
            if 'extracted_keywords' not in data:
                processed_answer = self.preprocessor.preprocess(data['reference_answer'])
                
                # Extract keywords if not already provided
                if 'keywords' not in data:
                    keywords = self.keyword_extractor.extract_keywords(processed_answer)
                    self.reference_data[question_id]['keywords'] = keywords
                    
                # Store processed answer for later use
                self.reference_data[question_id]['processed_answer'] = processed_answer
    
    def evaluate_answer(self, question_id, user_answer):
        """
        Evaluate a user's answer to a specific question.
        
        Args:
            question_id: Identifier for the question
            user_answer: User's answer text
            
        Returns:
            Score and detailed breakdown
        """
        if question_id not in self.reference_data:
            raise ValueError(f"Unknown question ID: {question_id}")
            
        # Get reference data
        ref_data = self.reference_data[question_id]
        ref_answer = ref_data['reference_answer']
        ref_keywords = ref_data.get('keywords', [])
        
        # Preprocess user answer
        processed_user_answer = self.preprocessor.preprocess(user_answer)
        
        # Extract keywords from user answer if needed
        user_keywords = self.keyword_extractor.extract_keywords(processed_user_answer)
        
        # Score the answer
        score, breakdown = self.scorer.score_answer(
            processed_user_answer,
            ref_data.get('processed_answer', ref_answer),
            ref_keywords
        )
        
        # Add user keywords to breakdown
        breakdown['user_keywords'] = user_keywords
        
        return score, breakdown
    
    def add_question(self, question_id, question, reference_answer, keywords=None):
        """
        Add a new question to the reference data.
        
        Args:
            question_id: Unique identifier for the question
            question: Question text
            reference_answer: Reference answer text
            keywords: Optional list of important keywords with weights
        """
        # Process reference answer
        processed_answer = self.preprocessor.preprocess(reference_answer)
        
        # Extract keywords if not provided
        if keywords is None:
            keywords = self.keyword_extractor.extract_keywords(processed_answer)
        
        # Add to reference data
        self.reference_data[question_id] = {
            'question': question,
            'reference_answer': reference_answer,
            'processed_answer': processed_answer,
            'keywords': keywords
        }
        
    def save_reference_data(self, filepath):
        """Save reference data to a JSON file."""
        # Convert any non-serializable data
        serializable_data = {}
        for q_id, data in self.reference_data.items():
            serializable_data[q_id] = {
                'question': data['question'],
                'reference_answer': data['reference_answer'],
                'keywords': data['keywords']
                # Note: We don't save processed_answer as it can be regenerated
            }
            
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
            
    def get_explanation(self, score, breakdown):
        """
        Generate a human-friendly explanation of the evaluation results.
        
        Args:
            score: The final score
            breakdown: The score breakdown
            
        Returns:
            Explanation text
        """
        # Create an explanation
        explanation = [f"Your answer received a score of {score:.1f}/100."]
        
        # Determine score category
        if score >= 90:
            explanation.append("This is an excellent answer that covers the key concepts thoroughly.")
        elif score >= 80:
            explanation.append("This is a very good answer that covers most of the important points.")
        elif score >= 70:
            explanation.append("This is a good answer that addresses the question adequately.")
        elif score >= 60:
            explanation.append("This is a satisfactory answer but could be improved with more detail.")
        elif score >= 50:
            explanation.append("This answer addresses some aspects of the question but needs significant improvement.")
        else:
            explanation.append("This answer needs substantial revision to address the question properly.")
        
        # Add component-specific feedback
        explanation.append("\nBreakdown of your score:")
        
        # Keyword coverage
        keyword_score = breakdown.get('keyword_score', 0)
        matched_keywords = breakdown.get('matched_keywords', [])
        explanation.append(f"- Keyword coverage: {keyword_score:.1f}/100")
        if keyword_score >= 80:
            explanation.append("  You included most of the important concepts.")
        elif keyword_score >= 60:
            explanation.append("  You included some important concepts, but missed others.")
        else:
            explanation.append("  Your answer is missing many key concepts.")
            
        # Show matched keywords (limit to 5 for readability)
        if matched_keywords:
            if len(matched_keywords) <= 5:
                explanation.append(f"  Included concepts: {', '.join(matched_keywords)}")
            else:
                explanation.append(f"  Some included concepts: {', '.join(matched_keywords[:5])}...")
                
        # Semantic similarity
        if 'semantic_score' in breakdown:
            semantic_score = breakdown['semantic_score']
            explanation.append(f"- Semantic similarity: {semantic_score:.1f}/100")
            if semantic_score >= 80:
                explanation.append("  Your answer closely matches the meaning of the reference answer.")
            elif semantic_score >= 60:
                explanation.append("  Your answer captures some of the meaning of the reference answer.")
            else:
                explanation.append("  Your answer differs significantly in meaning from the reference answer.")
        
        # Completeness
        completeness_score = breakdown.get('completeness_score', 0)
        explanation.append(f"- Completeness: {completeness_score:.1f}/100")
        if completeness_score >= 80:
            explanation.append("  Your answer is comprehensive and well-developed.")
        elif completeness_score >= 60:
            explanation.append("  Your answer has adequate depth but could include more detail.")
        else:
            explanation.append("  Your answer lacks sufficient depth and detail.")
        
        # Suggestions for improvement
        explanation.append("\nSuggestions for improvement:")
        
        # If keyword score is low
        if keyword_score < 70:
            explanation.append("- Include more key concepts and terminology in your answer.")
            
        # If semantic score is low
        if 'semantic_score' in breakdown and breakdown['semantic_score'] < 70:
            explanation.append("- Ensure your explanation captures the essential meaning of the concept.")
            
        # If completeness score is low
        if completeness_score < 70:
            explanation.append("- Develop your answer with more detail and examples.")
            
        # If order score is low
        if 'order_score' in breakdown and breakdown['order_score'] < 70:
            explanation.append("- Organize your answer to better reflect the logical flow of concepts.")
        
        return "\n".join(explanation)