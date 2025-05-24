#!/usr/bin/env python
"""
Script to create train/validation/test splits for NCERT data.
"""
import os
import sys
import argparse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_split import split_question_bank

def main():
    parser = argparse.ArgumentParser(description="Create train/validation/test splits for NCERT data")
    parser.add_argument("--data-dir", default="data", help="Directory containing NCERT data files")
    parser.add_argument("--output-dir", default=None, help="Directory to save splits (default: data/splits)")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio for training set")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Ratio for validation set")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Ratio for test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'splits')
    
    split_question_bank(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

if __name__ == "__main__":
    main()