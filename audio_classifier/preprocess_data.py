import os
import numpy as np
import torch
import pickle
import librosa
from tqdm import tqdm
import warnings
import argparse

# Import the extract_features function from cnn_lstm_classifier
from cnn_lstm_classifier import extract_features

# Suppress warnings
warnings.filterwarnings("ignore")

# Define paths
AUDIO_DIR = "../swda_audio"
PROCESSED_DIR = "./processed_data"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess_dataset(force_reprocess=False):
    """
    Extract features from all audio files and save them to disk.
    
    Args:
        force_reprocess: If True, reprocess all files even if they exist
    """
    # Check if processed data already exists
    spec_file = os.path.join(PROCESSED_DIR, "spectrograms.npy")
    features_file = os.path.join(PROCESSED_DIR, "features.npy")
    labels_file = os.path.join(PROCESSED_DIR, "labels.npy")
    
    if os.path.exists(spec_file) and os.path.exists(features_file) and not force_reprocess:
        print("\nProcessed data already exists. Use --force to reprocess.")
        print(f"Spectrograms: {spec_file}")
        print(f"Features: {features_file}")
        print(f"Labels: {labels_file}")
        return
    
    print("\n" + "="*80)
    print(f"PREPROCESSING AUDIO FILES")
    print("="*80)
    
    # Get audio files
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
    print(f"\nFound {len(audio_files)} audio files.")
    
    # Process audio files
    spectrograms = []
    feature_vectors = []
    labels = []
    
    print("\nExtracting spectrograms and features...")
    for audio_file in tqdm(audio_files):
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        spectrogram, feature_vector = extract_features(audio_path)
        
        if spectrogram is not None and feature_vector is not None:
            # Our simplified feature extraction should always return 11 features
            if len(feature_vector) == 11:
                spectrograms.append(spectrogram)
                feature_vectors.append(feature_vector)
                
                # Placeholder labeling logic - in reality, you would use actual labels
                # We'll use filename patterns as a heuristic for this demo
                if "_B_" in audio_file:
                    label = 1 if np.random.random() < 0.6 else 0  # 60% chance of being a question
                else:
                    label = 0 if np.random.random() < 0.8 else 1  # 80% chance of being a statement
                
                labels.append(label)
            else:
                print(f"Skipping {audio_file} due to unexpected feature count: {len(feature_vector)}")
    
    # Convert to numpy arrays
    spectrograms = np.array(spectrograms)
    feature_vectors = np.array(feature_vectors)
    labels = np.array(labels)
    
    print(f"\nProcessed {len(spectrograms)} valid audio files")
    print(f"Spectrogram shape: {spectrograms.shape}")
    print(f"Feature vector shape: {feature_vectors.shape}")
    print(f"Label distribution: {np.sum(labels)} questions, {len(labels) - np.sum(labels)} statements")
    
    # Save processed data
    print("\nSaving processed data...")
    np.save(os.path.join(PROCESSED_DIR, "spectrograms.npy"), spectrograms)
    np.save(os.path.join(PROCESSED_DIR, "features.npy"), feature_vectors)
    np.save(os.path.join(PROCESSED_DIR, "labels.npy"), labels)
    
    print("\n" + "="*80)
    print(f"PREPROCESSING COMPLETE")
    print(f"Data saved to {PROCESSED_DIR}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio data for training")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if files exist")
    args = parser.parse_args()
    
    preprocess_dataset(force_reprocess=args.force) 