"""
Main module for answer evaluation system.
"""
import json
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing import TextPreprocessor
from keyword_extraction import KeywordExtractor
from scoring import AnswerScorer
from evaluation import AnswerEvaluator
from semantic_analysis import SemanticAnalyzer

def initialize_system(config_file=None, use_spacy=True):
    """Initialize the answer evaluation system with optional config."""
    # Default configuration
    config = {
        "preprocessing": {
            "remove_stopwords": True,
            "lemmatize": True
        },
        "keyword_extraction": {
            "method": "frequency",
            "max_keywords": 15
        },
        "scoring": {
            "keyword_weight": 0.4,
            "keyword_order_weight": 0.1,
            "completeness_weight": 0.2,
            "semantic_weight": 0.3,
            "partial_match_threshold": 0.8
        },
        "semantic_analysis": {
            "model_name": "en_core_web_md",
            "similarity_method": "document"
        }
    }
    
    # Load custom config if provided
    if config_file:
        with open(config_file, 'r') as f:
            custom_config = json.load(f)
            # Update config with custom values
            for section, values in custom_config.items():
                if section in config:
                    config[section].update(values)
                else:
                    config[section] = values
    
    # Initialize components
    preprocessor = TextPreprocessor(**config["preprocessing"])
    keyword_extractor = KeywordExtractor(**config["keyword_extraction"])
    
    # Initialize semantic analyzer if requested
    semantic_analyzer = None
    if use_spacy:
        try:
            semantic_analyzer = SemanticAnalyzer(
                model_name=config["semantic_analysis"]["model_name"]
            )
            print("Semantic analyzer initialized.")
        except Exception as e:
            print(f"Error initializing semantic analyzer: {e}")
            print("Falling back to keyword-based scoring only.")
    
    # Initialize scorer with semantic analyzer
    scorer = AnswerScorer(
        **config["scoring"],
        semantic_analyzer=semantic_analyzer
    )
    
    # Initialize evaluator
    evaluator = AnswerEvaluator(
        preprocessor=preprocessor,
        keyword_extractor=keyword_extractor,
        scorer=scorer
    )
    
    return evaluator

def demo_evaluation():
    """Run a demonstration of the answer evaluation system."""
    evaluator = initialize_system()

    # Load data from files
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    with open(os.path.join(data_dir, 'questions.json')) as f:
        questions = json.load(f)
    with open(os.path.join(data_dir, 'reference_answers.json')) as f:
        reference_answers = json.load(f)
    with open(os.path.join(data_dir, 'keywords.json')) as f:
        keywords = json.load(f)
    with open(os.path.join(data_dir, 'sample_responses.json')) as f:
        user_answers = json.load(f)

    # Prepare reference data for evaluator
    reference_data = {}
    for q_id in questions:
        reference_data[q_id] = {
            "question": questions[q_id],
            "reference_answer": reference_answers[q_id],
            "keywords": keywords[q_id]
        }
    evaluator.reference_data = reference_data

    # Evaluate and print results
    for q_id, answers in user_answers.items():
        for answer in answers:
            score, breakdown = evaluator.evaluate_answer(q_id, answer)
            print(f"\nQuestion: {reference_data[q_id]['question']}")
            print(f"Reference Answer: {reference_data[q_id]['reference_answer']}")
            print(f"User Answer: {answer}")
            print(f"Score: {score:.1f}/100")
            print("Breakdown:")
            print(f"  - Keyword Score: {breakdown['keyword_score']:.1f}/100")
            print(f"  - Order Score: {breakdown['order_score']:.1f}/100")
            print(f"  - Completeness Score: {breakdown['completeness_score']:.1f}/100")
            
            # Print semantic scores if available
            if 'semantic_score' in breakdown and breakdown['semantic_score'] > 0:
                print(f"  - Semantic Score: {breakdown['semantic_score']:.1f}/100")
                print(f"  - Concept Coverage: {breakdown['concept_coverage_score']:.1f}/100")
            
            print(f"  - Matched Keywords: {', '.join(breakdown['matched_keywords'])}")
            
            # Print matched concepts if available
            if 'matched_concepts' in breakdown and breakdown['matched_concepts']:
                print(f"  - Matched Concepts: {', '.join(breakdown['matched_concepts'][:10])}")
                if len(breakdown['matched_concepts']) > 10:
                    print(f"    ...and {len(breakdown['matched_concepts']) - 10} more")
            
            print(f"  - User Keywords: {', '.join(breakdown['user_keywords'])}")

def ncert_demo():
    """Run a demonstration using the NCERT question bank."""
    evaluator = initialize_system()

    # Load NCERT data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Load both question banks
    with open(os.path.join(data_dir, 'question_bank_NCERT_part1.json'), encoding='utf-8') as f:
        question_bank1 = json.load(f)
    with open(os.path.join(data_dir, 'question_bank_NCERT_part2.json'), encoding='utf-8') as f:
        question_bank2 = json.load(f)

    # Combine both into a single dictionary
    question_bank = {}
    question_bank.update(question_bank1)
    question_bank.update(question_bank2)
    
    # Select a random subject and question
    import random
    subjects = list(question_bank.keys())
    subject = random.choice(subjects)
    questions = question_bank[subject]
    question_id = random.choice(list(questions.keys()))
    
    question_data = questions[question_id]
    
    print(f"Subject: {subject}")
    print(f"Question: {question_data['question']}")
    print("\nPlease provide your answer (or type 'exit' to quit):")
    user_answer = input()
    
    if user_answer.lower() == 'exit':
        return
    
    # Prepare reference data
    reference_data = {
        f"{subject}_{question_id}": {
            "question": question_data["question"],
            "reference_answer": question_data["reference_answer"],
            "keywords": question_data["keywords"]
        }
    }
    evaluator.reference_data = reference_data
    
    # Evaluate the answer
    score, breakdown = evaluator.evaluate_answer(f"{subject}_{question_id}", user_answer)
    
    print("\nReference Answer:")
    print(question_data["reference_answer"])
    
    print("\nYour Answer:")
    print(user_answer)
    
    print(f"\nScore: {score:.1f}/100")
    print("Breakdown:")
    print(f"  - Keyword Score: {breakdown['keyword_score']:.1f}/100")
    print(f"  - Order Score: {breakdown['order_score']:.1f}/100")
    print(f"  - Completeness Score: {breakdown['completeness_score']:.1f}/100")
    
    # Print semantic scores if available
    if 'semantic_score' in breakdown and breakdown['semantic_score'] > 0:
        print(f"  - Semantic Score: {breakdown['semantic_score']:.1f}/100")
        print(f"  - Concept Coverage: {breakdown['concept_coverage_score']:.1f}/100")
    
    print(f"  - Matched Keywords: {', '.join(breakdown['matched_keywords'])}")
    
    # Print matched concepts if available
    if 'matched_concepts' in breakdown and breakdown['matched_concepts']:
        print(f"  - Matched Concepts: {', '.join(breakdown['matched_concepts'][:10])}")
        if len(breakdown['matched_concepts']) > 10:
            print(f"    ...and {len(breakdown['matched_concepts']) - 10} more")
    
    print(f"  - User Keywords: {', '.join(breakdown['user_keywords'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer Evaluation System")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--ncert", action="store_true", help="Run NCERT demonstration")
    parser.add_argument("--no-spacy", action="store_true", help="Disable spaCy semantic analysis")
    args = parser.parse_args()
    
    if args.demo:
        demo_evaluation()
    elif args.ncert:
        ncert_demo()
    else:
        # Initialize system
        evaluator = initialize_system(args.config, use_spacy=not args.no_spacy)
        print("Answer evaluation system initialized.")
        print("Use --demo or --ncert to run a demonstration.")