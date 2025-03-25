"""
Bridge script to extract features from data_cleaning audio files and prepare them for training.
"""

import os
import argparse
import json
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from feature_extractor import MelSpectrogramExtractor, ProsodyFeatureExtractor, Wav2Vec2FeatureExtractor

def create_dataset_splits(audio_files, labels, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        audio_files (list): List of audio file paths
        labels (list): List of corresponding labels
        train_size (float): Proportion of data for training
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing the splits
    """
    # First split: train vs. (val + test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        audio_files, labels, test_size=(val_size + test_size), random_state=random_state, stratify=labels
    )
    
    # Second split: val vs. test
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, 
        test_size=test_size/(val_size + test_size), 
        random_state=random_state,
        stratify=temp_labels
    )
    
    return {
        'train': (train_files, train_labels),
        'dev': (val_files, val_labels),
        'test': (test_files, test_labels)
    }

def extract_and_save_features(audio_dir, output_dir, dataset_splits, feature_types=None, device=None):
    """
    Extract features from audio files and save them to disk.
    
    Args:
        audio_dir (str): Directory containing audio files
        output_dir (str): Directory to save processed features
        dataset_splits (dict): Dictionary containing dataset splits
        feature_types (list): List of feature types to extract
        device (str): Device to use for feature extraction
    """
    if feature_types is None:
        feature_types = ['mel_spectrogram', 'prosody']
    
    # Initialize feature extractors
    extractors = {}
    if 'mel_spectrogram' in feature_types:
        extractors['mel_spectrogram'] = MelSpectrogramExtractor()
    if 'prosody' in feature_types:
        extractors['prosody'] = ProsodyFeatureExtractor()
    if 'wav2vec2' in feature_types:
        extractors['wav2vec2'] = Wav2Vec2FeatureExtractor(device=device)
    
    # Process each split
    for split, (files, labels) in dataset_splits.items():
        print(f"Processing {split} split ({len(files)} files)...")
        
        # Create output directory for this split
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Process each file
        for i, (file_path, label) in enumerate(tqdm(zip(files, labels), total=len(files))):
            # Load audio
            try:
                audio_path = os.path.join(audio_dir, file_path)
                waveform, sample_rate = torchaudio.load(audio_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
            
            # Extract features
            features = {}
            for feature_type, extractor in extractors.items():
                try:
                    if feature_type == 'mel_spectrogram':
                        features[feature_type] = extractor(waveform)
                    else:
                        features[feature_type] = extractor(waveform, sample_rate)
                except Exception as e:
                    print(f"Error extracting {feature_type} features for {file_path}: {e}")
                    if feature_type == 'mel_spectrogram':
                        # Create a dummy feature as a fallback
                        features[feature_type] = torch.zeros((128, 100))
                    else:
                        features[feature_type] = torch.zeros(12)
            
            # Save features
            output_path = os.path.join(split_dir, f"{i:05d}.pt")
            torch.save({
                'file_id': file_path,
                'features': features,
                'label': label,
                'sample_rate': sample_rate
            }, output_path)
    
    print("Feature extraction complete!")

def create_metadata_files(dataset_splits, output_dir, label_map=None):
    """
    Create metadata JSON files for each split.
    
    Args:
        dataset_splits (dict): Dictionary containing dataset splits
        output_dir (str): Directory to save metadata files
        label_map (dict): Dictionary mapping label indices to label names
    """
    if label_map is None:
        label_map = {0: 'non-qy^d', 1: 'qy^d'}
    
    # Create metadata for each split
    for split, (files, labels) in dataset_splits.items():
        metadata = {}
        for i, (file_path, label) in enumerate(zip(files, labels)):
            metadata[f"{i:05d}"] = {
                'original_path': file_path,
                'label': int(label),
                'label_name': label_map[int(label)]
            }
        
        # Save metadata
        with open(os.path.join(output_dir, f"{split}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print("Metadata files created!")

def process_data_cleaning_output(args):
    """
    Process the data_cleaning output files and prepare them for training.
    
    Args:
        args: Command line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Collect all audio files from data_cleaning
    audio_files = []
    labels = []
    
    # Process switchboard corpus data from CSV
    if os.path.exists(args.csv_path):
        print("Processing Switchboard corpus from CSV...")
        df = pd.read_csv(args.csv_path)
        
        # Filter out rows without audio files
        df = df[df['found'].notna() & (df['found'] != '')]
        
        # Extract audio files and labels
        for _, row in df.iterrows():
            # Handle multiple audio files separated by commas
            if isinstance(row['found'], str) and row['found'].strip():
                file_paths = [path.strip() for path in row['found'].split(',') if path.strip()]
                for path in file_paths:
                    if os.path.exists(os.path.join(args.audio_dir, path)):
                        audio_files.append(path)
                        # Extract label
                        if 'qy^d' in str(row['meta.tag']):
                            labels.append(1)  # Declarative question
                        else:
                            labels.append(0)  # Not a declarative question
    
    # Process MRDA corpus data by scanning directories
    print("Processing MRDA corpus by scanning directories...")
    for root, dirs, files in os.walk(args.audio_dir):
        for file in files:
            if file.endswith('_audio.wav'):
                rel_path = os.path.relpath(os.path.join(root, file), args.audio_dir)
                # Check if this file is already in the list
                if rel_path not in audio_files:
                    audio_files.append(rel_path)
                    # For MRDA corpus, all extracted files are declarative questions (qy^d)
                    labels.append(1)
    
    print(f"Total audio files: {len(audio_files)}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Step 2: Split the dataset
    dataset_splits = create_dataset_splits(audio_files, labels)
    
    # Step 3: Extract features and save them
    extract_and_save_features(
        args.audio_dir, 
        args.output_dir, 
        dataset_splits, 
        feature_types=args.feature_types,
        device=args.device
    )
    
    # Step 4: Create metadata files
    create_metadata_files(dataset_splits, args.output_dir)
    
    print(f"Processed data saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data_cleaning output for training")
    
    parser.add_argument("--audio_dir", type=str, default="../data_cleaning/split_audiov2",
                        help="Directory containing audio files")
    parser.add_argument("--csv_path", type=str, default="../data_cleaning/swda_declarative.csv",
                        help="Path to the swda_declarative.csv file")
    parser.add_argument("--output_dir", type=str, default="../data/processed",
                        help="Directory to save processed features")
    parser.add_argument("--feature_types", nargs='+', 
                        default=['mel_spectrogram', 'prosody'],
                        help="Types of features to extract")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for feature extraction")
    
    args = parser.parse_args()
    
    process_data_cleaning_output(args) 