# Answer Evaluation System

An NLP-based system for evaluating student answers to science questions using semantic analysis and keyword matching.

## Overview

This system uses natural language processing techniques to evaluate free-text student answers against reference answers from class 5 to class 10 NCERT textbooks. It combines keyword matching, semantic similarity analysis, and text processing to provide comprehensive scoring and feedback.

## Features

- Evaluates free-text answers against reference answers from NCERT textbooks
- Uses both keyword matching and semantic understanding for scoring
- Provides detailed feedback with score breakdowns
- Includes a Streamlit web interface for interactive evaluation
- Pre-loaded with NCERT science questions from classes 5-10 across multiple subjects
- Visualization of evaluation metrics
- Support for both simple keyword-based and advanced semantic analysis

## System Components

- **Preprocessing**: Tokenization, lemmatization, and stopword removal
- **Keyword Extraction**: Identifies important keywords in student answers
- **Semantic Analysis**: Measures semantic similarity between student and reference answers
- **Scoring Engine**: Combines multiple metrics for comprehensive evaluation
- **Web Interface**: Interactive evaluation through Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AkhilMittapalli/NLP-Based-Automated-Answering-System.git
```

2. Set up a virtual environment:

**Windows (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK and spaCy resources:
```bash
python setup.py
```

## Usage

### Running the Web Interface

```bash
streamlit run app.py
```

This will launch the Streamlit web application where you can:
- Select a subject/topic
- Answer randomly selected questions
- View detailed evaluation of your answers with score breakdowns

## Project Structure

- `app.py`: Streamlit web interface
- `evaluation.py`: Main evaluation system
- `preprocessing.py`: Text preprocessing utilities
- `keyword_extraction.py`: Keyword extraction and matching
- `scoring.py`: Answer scoring logic
- `semantic_analysis.py`: Semantic similarity analysis using spaCy
- `main.py`: System initialization and demo functions
- `setup.py`: Script to download required NLTK and spaCy resources
- `config.json`: Configuration settings for the evaluation system
- `data/`: NCERT question banks and reference answers

## Configuration

The system can be configured through `config.json`, with options for:
- Preprocessing settings (stopword removal, lemmatization)
- Keyword extraction method and parameters
- Scoring weights for different evaluation components
- Semantic analysis model and methods

## Requirements

- Python 3.7+
- NLTK 3.7.0+
- scikit-learn 1.1.0+
- spaCy 3.5.0+
- Streamlit 1.20.0+
- Matplotlib 3.5.0+
- NumPy 1.21.0+
