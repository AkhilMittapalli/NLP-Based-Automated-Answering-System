#!/usr/bin/env python
"""
Script to evaluate answer evaluation system on different data splits.
"""
import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from main import initialize_system
from data_split import load_split, prepare_reference_data

def evaluate_system(evaluator, data, output_file=None):
    """
    Evaluate the system on a data split.
    
    Args:
        evaluator: Initialized evaluator
        data: Data to evaluate on
        output_file: File to save results to
        
    Returns:
        Dictionary of results
    """
    # Prepare data for evaluation
    reference_data = prepare_reference_data(data)
    evaluator.reference_data = reference_data
    
    # Evaluation metrics
    results = {
        "scores": [],
        "keyword_scores": [],
        "semantic_scores": [],
        "completeness_scores": [],
        "by_subject": {}
    }
    
    # Create simulated user answers of various qualities
    qualities = {
        "high": 0.9,  # 90% of reference answer
        "medium": 0.6,  # 60% of reference answer
        "low": 0.3,  # 30% of reference answer
    }
    
    # Evaluate each question with simulated answers
    for subject, questions in tqdm(data.items(), desc="Evaluating subjects"):
        subject_results = {q: [] for q in qualities.keys()}
        
        for q_id, q_data in tqdm(questions.items(), desc=f"Subject: {subject}", leave=False):
            full_id = f"{subject}_{q_id}"
            ref_answer = q_data["reference_answer"]
            
            # Create simulated answers by taking portions of the reference answer
            words = ref_answer.split()
            for quality_name, quality_factor in qualities.items():
                n_words = max(1, int(len(words) * quality_factor))
                simulated_answer = " ".join(words[:n_words])
                
                # Evaluate the answer
                try:
                    score, breakdown = evaluator.evaluate_answer(full_id, simulated_answer)
                    
                    # Store scores
                    results["scores"].append(score)
                    results["keyword_scores"].append(breakdown["keyword_score"])
                    results["semantic_scores"].append(breakdown.get("semantic_score", 0))
                    results["completeness_scores"].append(breakdown["completeness_score"])
                    
                    # Store by subject
                    subject_results[quality_name].append(score)
                except Exception as e:
                    print(f"Error evaluating {full_id}: {e}")
        
        # Calculate subject averages
        results["by_subject"][subject] = {
            quality: np.mean(scores) if scores else 0 
            for quality, scores in subject_results.items()
        }
    
    # Calculate overall metrics
    for key in ["scores", "keyword_scores", "semantic_scores", "completeness_scores"]:
        if results[key]:
            results[f"avg_{key}"] = np.mean(results[key])
            results[f"std_{key}"] = np.std(results[key])
            results[f"min_{key}"] = min(results[key])
            results[f"max_{key}"] = max(results[key])
    
    # Calculate quality-specific metrics
    quality_scores = {quality: [] for quality in qualities.keys()}
    for subject_results in results["by_subject"].values():
        for quality, score in subject_results.items():
            quality_scores[quality].append(score)
    
    for quality, scores in quality_scores.items():
        if scores:
            results[f"avg_{quality}"] = np.mean(scores)
            results[f"std_{quality}"] = np.std(scores)
    
    # Save results if output file provided
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate answer evaluation system on data splits")
    parser.add_argument("--data-dir", default="data", help="Directory containing data")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Data split to evaluate on")
    parser.add_argument("--output-dir", default="results", help="Directory to save results")
    parser.add_argument("--no-spacy", action="store_true", help="Disable spaCy semantic analysis")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = initialize_system(use_spacy=not args.no_spacy)
    
    # Load data
    print(f"Loading {args.split} data...")
    data = load_split(args.data_dir, args.split)
    
    # Evaluate
    print("Evaluating system...")
    output_file = os.path.join(args.output_dir, f"{args.split}_results.json")
    results = evaluate_system(evaluator, data, output_file)
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Average score: {results.get('avg_scores', 0):.2f}")
    print(f"Average scores by quality:")
    for quality in ["high", "medium", "low"]:
        print(f"  {quality.capitalize()}: {results.get(f'avg_{quality}', 0):.2f}")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()