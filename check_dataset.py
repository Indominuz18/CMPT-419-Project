#!/usr/bin/env python
import os
import json
import numpy as np
import argparse
from src.data_loader import SLUEDataset

def main():
    parser = argparse.ArgumentParser(description='Check dataset label distribution')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory containing processed data')
    args = parser.parse_args()
    
    # Load metadata
    splits = ['train', 'val', 'test']
    for split in splits:
        metadata_path = os.path.join(args.data_dir, f'{split}_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Extract labels - metadata is a dictionary with keys like '00000', '00001', etc.
            labels = [item['label'] for item in metadata.values()]
            unique_labels = np.unique(labels)
            label_counts = np.bincount(labels)
            
            print(f"\n{split.upper()} SET:")
            print(f"Number of samples: {len(labels)}")
            print(f"Unique labels: {unique_labels}")
            print(f"Label distribution: {label_counts}")
            
            if len(unique_labels) <= 1:
                print(f"WARNING: {split} set contains only {len(unique_labels)} class!")
            
            # Calculate class percentages
            for label, count in enumerate(label_counts):
                if label < len(label_counts):
                    percentage = (count / len(labels)) * 100
                    print(f"Class {label}: {count} samples ({percentage:.2f}%)")
        else:
            print(f"No metadata found for {split} split at {metadata_path}")
    
    print("\nLoading datasets to check if they load correctly...")
    try:
        for split in splits:
            dataset = SLUEDataset(args.data_dir, split)
            print(f"{split} dataset size: {len(dataset)}")
    except Exception as e:
        print(f"Error loading datasets: {e}")

if __name__ == "__main__":
    main() 