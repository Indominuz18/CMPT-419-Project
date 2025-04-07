#!/usr/bin/env python
import os
import argparse
import subprocess
import sys
import torch

def check_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs('processed_data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def run_feature_extraction():
    """Run the feature extraction stage"""
    print("=== STAGE 1: FEATURE EXTRACTION ===")
    from feature_extractor import process_dataset
    stats = process_dataset()
    print("\nDataset Statistics:")
    print(f"Total samples: {stats['total']}")
    print(f"Questions: {stats['questions']} ({stats['question_ratio']:.2%})")
    print(f"Statements: {stats['statements']} ({1-stats['question_ratio']:.2%})")

def run_model_training():
    """Run the model training stage"""
    print("=== STAGE 2: MODEL TRAINING ===")
    from model_trainer import train_model
    
    # Detect and display available devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = train_model()
    print(f"\nTraining completed. Best model saved with F1 score: {results['best_f1_score']:.4f}")
    print(f"Model saved to: {results['model_path']}")

def run_demo():
    """Run the demo application"""
    print("=== STAGE 3: DEMO APPLICATION ===")
    subprocess.run([sys.executable, 'demo_app.py'])

def main():
    parser = argparse.ArgumentParser(
        description='Audio Classification: Question vs. Statement'
    )
    parser.add_argument(
        'stage', type=str, choices=['extract', 'train', 'demo', 'all'],
        help='Which stage to run: extract (feature extraction), train (model training), demo (GUI application), or all (run all stages)'
    )
    parser.add_argument(
        '--cpu', action='store_true',
        help='Force CPU usage even if CUDA is available'
    )
    
    args = parser.parse_args()
    
    # Set device based on arguments
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Forcing CPU usage as requested")
    
    # Create necessary directories
    check_directories()
    
    # Run the selected stage
    if args.stage == 'extract' or args.stage == 'all':
        run_feature_extraction()
    
    if args.stage == 'train' or args.stage == 'all':
        run_model_training()
    
    if args.stage == 'demo' or args.stage == 'all':
        run_demo()

if __name__ == '__main__':
    main() 