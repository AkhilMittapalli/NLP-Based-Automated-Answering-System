"""
Data splitting functionality for NCERT answer evaluation system.
"""
import json
import random
import os
from collections import defaultdict

def split_question_bank(data_dir, output_dir=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split the question bank into train, validation, and test sets.
    
    Args:
        data_dir: Directory containing question bank JSON files
        output_dir: Directory to save split data files (default: data/splits)
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing train, val, and test splits
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1.0"
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'splits')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all question bank files
    question_bank = {}
    for i in range(1, 6):  # Assuming there could be up to 5 parts
        file_path = os.path.join(data_dir, f'question_bank_NCERT_part{i}.json')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                part_data = json.load(f)
                question_bank.update(part_data)
    
    # Organize questions by subject to ensure balanced splits
    subject_questions = defaultdict(list)
    for subject, questions in question_bank.items():
        for q_id, q_data in questions.items():
            subject_questions[subject].append((q_id, q_data))
    
    # Create empty splits
    splits = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    # Split each subject proportionally
    for subject, questions in subject_questions.items():
        # Shuffle questions
        random.shuffle(questions)
        
        # Calculate split sizes
        n_questions = len(questions)
        n_train = int(n_questions * train_ratio)
        n_val = int(n_questions * val_ratio)
        
        # Split the questions
        train_questions = questions[:n_train]
        val_questions = questions[n_train:n_train + n_val]
        test_questions = questions[n_train + n_val:]
        
        # Add to respective splits
        if subject not in splits['train']:
            splits['train'][subject] = {}
        if subject not in splits['val']:
            splits['val'][subject] = {}
        if subject not in splits['test']:
            splits['test'][subject] = {}
            
        for q_id, q_data in train_questions:
            splits['train'][subject][q_id] = q_data
        for q_id, q_data in val_questions:
            splits['val'][subject][q_id] = q_data
        for q_id, q_data in test_questions:
            splits['test'][subject][q_id] = q_data
    
    # Save the splits
    for split_name, split_data in splits.items():
        output_path = os.path.join(output_dir, f'{split_name}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2)
    
    # Print statistics
    train_count = sum(len(questions) for subject in splits['train'].values() for questions in [subject])
    val_count = sum(len(questions) for subject in splits['val'].values() for questions in [subject])
    test_count = sum(len(questions) for subject in splits['test'].values() for questions in [subject])
    total = train_count + val_count + test_count
    
    print(f"Split complete. Total questions: {total}")
    print(f"Train: {train_count} ({train_count/total:.1%})")
    print(f"Validation: {val_count} ({val_count/total:.1%})")
    print(f"Test: {test_count} ({test_count/total:.1%})")
    print(f"Splits saved to {output_dir}")
    
    return splits

def load_split(data_dir, split_name):
    """
    Load a specific data split.
    
    Args:
        data_dir: Directory containing the data
        split_name: Name of the split to load ('train', 'val', or 'test')
        
    Returns:
        Dictionary with the split data
    """
    split_path = os.path.join(data_dir, 'splits', f'{split_name}.json')
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found at {split_path}. Run split_question_bank first.")
        
    with open(split_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_reference_data(split_data):
    """
    Convert split data format to reference data format for the evaluator.
    
    Args:
        split_data: Data in the split format
        
    Returns:
        Dictionary in reference_data format for the evaluator
    """
    reference_data = {}
    for subject, questions in split_data.items():
        for q_id, q_data in questions.items():
            # Create a unique ID combining subject and question ID
            full_id = f"{subject}_{q_id}"
            reference_data[full_id] = {
                "question": q_data["question"],
                "reference_answer": q_data["reference_answer"],
                "keywords": q_data["keywords"]
            }
    return reference_data