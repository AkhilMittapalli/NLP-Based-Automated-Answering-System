"""
Text preprocessing utilities for answer evaluation system.
"""
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    """Handle text preprocessing tasks for answer evaluation."""
    
    def __init__(self, remove_stopwords=True, lemmatize=True, lowercase=True):
        """Initialize the text preprocessor with specified options."""
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    
    def preprocess(self, text):
        """Process text by tokenizing, removing punctuation, etc."""
        if not text or not isinstance(text, str):
            return []
        
        # Lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation and tokenize
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        tokens = word_tokenize(text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Lemmatize if specified
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        # Remove any empty tokens
        tokens = [t for t in tokens if t.strip()]
        
        return tokens
    
    def preprocess_keep_sentences(self, text):
        """Preprocess text but maintain sentence structure."""
        sentences = nltk.sent_tokenize(text)
        processed_sentences = []
        
        for sentence in sentences:
            processed_tokens = self.preprocess(sentence)
            if processed_tokens:
                processed_sentences.append(processed_tokens)
                
        return processed_sentences