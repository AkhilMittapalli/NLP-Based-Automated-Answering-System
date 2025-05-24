"""
Setup script to download required NLTK and spaCy models.
"""
import nltk
import os
import subprocess
import sys

def download_nltk_resources():
    """Download required NLTK resources."""
    print("Downloading NLTK resources...")
    
    resources = [
        'punkt',
        'stopwords',
        'wordnet'
    ]
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"  ✓ Downloaded {resource}")
        except Exception as e:
            print(f"  ✗ Failed to download {resource}: {e}")

def download_spacy_models():
    """Download required spaCy models."""
    print("\nDownloading spaCy models...")
    
    models = [
        'en_core_web_md'  # Medium-sized English model
    ]
    
    for model in models:
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            print(f"  ✓ Downloaded {model}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to download {model}: {e}")

def main():
    """Main setup function."""
    print("Setting up NCERT Answer Evaluation System...")
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Download spaCy models
    download_spacy_models()
    
    print("\nSetup complete!")

if __name__ == "__main__":
    main()