import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import argparse
import warnings

# Import the model from cnn_lstm_classifier
from cnn_lstm_classifier import CNNLSTMClassifier, train_model

# Suppress warnings
warnings.filterwarnings("ignore")

# Define paths
PROCESSED_DIR = "./processed_data"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class for spectrogram + features
class AudioSpectrogramDataset(Dataset):
    def __init__(self, spectrograms, features, labels):
        self.spectrograms = spectrograms
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.spectrograms[idx],
            self.features[idx],
            self.labels[idx]
        )

def train_cnn_lstm_model(epochs=50, batch_size=None, learning_rate=0.001):
    """
    Train the CNN+LSTM model using preprocessed data.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training (default: auto-select based on GPU)
        learning_rate: Learning rate for optimizer
    """
    print("\n" + "="*80)
    print(f"TRAINING CNN+LSTM MODEL")
    print(f"Using device: {device}")
    print("="*80)
    
    # Check if preprocessed data exists
    spec_file = os.path.join(PROCESSED_DIR, "spectrograms.npy")
    features_file = os.path.join(PROCESSED_DIR, "features.npy")
    labels_file = os.path.join(PROCESSED_DIR, "labels.npy")
    
    if not (os.path.exists(spec_file) and os.path.exists(features_file) and os.path.exists(labels_file)):
        print("\nPreprocessed data not found. Please run preprocess_data.py first.")
        return
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    spectrograms = np.load(spec_file)
    feature_vectors = np.load(features_file)
    labels = np.load(labels_file)
    
    print(f"Loaded {len(spectrograms)} processed audio files")
    print(f"Spectrogram shape: {spectrograms.shape}")
    print(f"Feature vector shape: {feature_vectors.shape}")
    print(f"Label distribution: {np.sum(labels)} questions, {len(labels) - np.sum(labels)} statements")
    
    # Scale feature vectors
    print("\nScaling features...")
    scaler = StandardScaler()
    feature_vectors_scaled = scaler.fit_transform(feature_vectors)
    
    # Add channel dimension to spectrograms for CNN
    spectrograms = spectrograms.reshape(spectrograms.shape[0], 1, spectrograms.shape[1], spectrograms.shape[2])
    
    # Split data
    print("\nSplitting data into train/test sets...")
    X_spec_train, X_spec_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
        spectrograms, feature_vectors_scaled, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Convert to PyTorch tensors
    X_spec_train = torch.FloatTensor(X_spec_train)
    X_spec_test = torch.FloatTensor(X_spec_test)
    X_feat_train = torch.FloatTensor(X_feat_train)
    X_feat_test = torch.FloatTensor(X_feat_test)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    y_test = torch.FloatTensor(y_test).view(-1, 1)
    
    # Create datasets and dataloaders
    train_dataset = AudioSpectrogramDataset(X_spec_train, X_feat_train, y_train)
    test_dataset = AudioSpectrogramDataset(X_spec_test, X_feat_test, y_test)
    
    # Using larger batch size for GPU acceleration
    if batch_size is None:
        batch_size = 32 if torch.cuda.is_available() else 16
    
    print(f"\nCreating data loaders with batch size {batch_size}...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4 if torch.cuda.is_available() else 0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4 if torch.cuda.is_available() else 0
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = CNNLSTMClassifier(n_features=X_feat_train.shape[1], n_mels=spectrograms.shape[2])
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Train model
    print("\nStarting model training...")
    train_losses, test_losses, accuracies = train_model(
        train_loader, test_loader, model, criterion, optimizer, device, num_epochs=epochs
    )
    
    # Move model back to CPU for saving (to make it compatible with CPU inference)
    model = model.to('cpu')
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "cnn_lstm_model.pth"))
    with open(os.path.join(OUTPUT_DIR, "cnn_lstm_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    # Plot training metrics
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cnn_lstm_training.png"))
    
    print(f"Model saved to {os.path.join(OUTPUT_DIR, 'cnn_lstm_model.pth')}")
    print(f"Final test accuracy: {accuracies[-1]:.4f}")
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN+LSTM model on preprocessed data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    
    train_cnn_lstm_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    ) 