import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import random
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define paths
AUDIO_DIR = "../swda_audio"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to extract spectrograms and features from audio
def extract_features(audio_path, n_mfcc=13, n_mels=128, n_fft=2048, hop_length=512):
    """Extract spectrogram and features from an audio file."""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Generate mel spectrogram (for CNN input)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Fixed set of features (scalar values only)
        features = []
        
        # 1. MFCC statistics (mean and std only)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        features.append(np.mean(mfcc))
        features.append(np.std(mfcc))
        
        # 2. Chroma features (12 pitch classes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.append(np.mean(chroma))
        features.append(np.std(chroma))
        
        # 3. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.append(np.mean(spectral_centroid))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features.append(np.mean(spectral_bandwidth))
        
        # 4. Pitch features (critical for questions)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Clean up NaN values in pitch data
        f0_clean = f0[~np.isnan(f0)]
        
        # Set defaults if no valid pitch data
        f0_mean = 0
        f0_std = 0
        f0_trend = 0
        
        if len(f0_clean) > 10:  # Ensure enough valid points
            f0_mean = np.mean(f0_clean)
            f0_std = np.std(f0_clean)
            
            # Calculate pitch trend (rising/falling at the end)
            # Get the last 25% of the pitch contour
            last_quarter_idx = max(1, int(len(f0_clean) * 0.75))
            last_quarter = f0_clean[last_quarter_idx:]
            
            if len(last_quarter) > 1:
                # Fit a line to the last quarter of pitch values
                x = np.arange(len(last_quarter))
                try:
                    slope = np.polyfit(x, last_quarter, 1)[0]
                    f0_trend = slope
                except:
                    f0_trend = 0
        
        # Add pitch features
        features.append(f0_mean)
        features.append(f0_std)
        features.append(f0_trend)
        
        # 5. Energy (volume)
        rms = librosa.feature.rms(y=y)[0]
        features.append(np.mean(rms))
        
        # 6. Zero crossing rate (related to noisiness/voice quality)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.append(np.mean(zcr))
        
        # Handle varying lengths by padding/trimming
        target_length = sr * 3  # Target 3 seconds of audio
        if len(y) < target_length:
            # Pad audio if shorter than target length
            y_padded = np.pad(y, (0, target_length - len(y)), 'constant')
        else:
            # Trim audio if longer than target length
            y_padded = y[:target_length]
        
        # Regenerate spectrogram with padded/trimmed audio
        mel_spec_padded = librosa.feature.melspectrogram(
            y=y_padded, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        log_mel_spec_padded = librosa.power_to_db(mel_spec_padded, ref=np.max)
        
        # Convert list to numpy array of floats
        feature_vector = np.array(features, dtype=np.float32)
        
        return log_mel_spec_padded, feature_vector
    
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None, None

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

# Define CNN+LSTM model architecture
class CNNLSTMClassifier(nn.Module):
    def __init__(self, n_features, n_mels=128):
        super(CNNLSTMClassifier, self).__init__()
        
        # CNN layers for spectrogram processing
        self.cnn = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate the CNN output size
        # After 4 max pooling layers with stride 2, the dimensions are reduced by 2^4 = 16
        self.cnn_output_height = n_mels // 16
        
        # LSTM layer for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=256 * self.cnn_output_height,  # CNN output features
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Dense layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(512 + n_features, 256),  # 512 = bidirectional LSTM (256*2), plus additional features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, spectrogram, features):
        # Input spectrogram shape: [batch_size, 1, n_mels, time]
        batch_size = spectrogram.size(0)
        
        # Pass through CNN layers
        x = self.cnn(spectrogram)
        
        # Reshape for LSTM: [batch_size, time, features]
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, x.size(1), -1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use the last time step output
        lstm_out = lstm_out[:, -1, :]
        
        # Concatenate LSTM output with additional features
        combined = torch.cat((lstm_out, features), dim=1)
        
        # Pass through classifier
        output = self.classifier(combined)
        
        return output

# Function to train the model
def train_model(train_loader, test_loader, model, criterion, optimizer, device, num_epochs=100):
    train_losses = []
    test_losses = []
    accuracies = []
    
    # Move model to GPU
    model = model.to(device)
    
    print(f"\n{'='*80}")
    print(f"TRAINING STARTED - Using {device}")
    print(f"{'='*80}\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        print(f"\n{'-'*80}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'-'*80}")
        
        # Training progress
        print("Training:")
        for batch_idx, (specs, feats, labels) in enumerate(train_loader):
            # Move tensors to the device
            specs = specs.to(device)
            feats = feats.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(specs, feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy for this batch
            predicted = (outputs >= 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Print progress every 5 batches
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        print("\nEvaluation:")
        with torch.no_grad():
            for specs, feats, labels in test_loader:
                # Move tensors to the device
                specs = specs.to(device)
                feats = feats.to(device)
                labels = labels.to(device)
                
                outputs = model(specs, feats)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_losses.append(test_loss)
        accuracy = correct / total
        accuracies.append(accuracy)
        
        # Extract predictions for questions and statements
        question_accuracy = 0
        statement_accuracy = 0
        if total > 0:
            # This is a simplified version - in real code, you'd need to compute this from actual predictions
            question_count = int(sum(labels.cpu().numpy()))
            statement_count = total - question_count
            
            if question_count > 0:
                question_accuracy = correct / total  # Simplified, ideally calculate specifically for questions
            if statement_count > 0:
                statement_accuracy = correct / total  # Simplified, ideally calculate specifically for statements
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Training Loss:   {train_loss:.4f}")
        print(f"  Training Acc:    {train_accuracy:.4f}")
        print(f"  Validation Loss: {test_loss:.4f}")
        print(f"  Validation Acc:  {accuracy:.4f}")
        
        # Print learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate:   {current_lr:.6f}")
        
        if epoch > 0:
            loss_change = test_losses[-2] - test_losses[-1]
            acc_change = accuracies[-1] - accuracies[-2]
            print(f"  Loss Change:     {loss_change:.4f}")
            print(f"  Accuracy Change: {acc_change:.4f}")
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED - Final Accuracy: {accuracies[-1]:.4f}")
    print(f"{'='*80}\n")
    
    return train_losses, test_losses, accuracies

# Main function
def main():
    print("\n========================================================================")
    print("Starting CNN+LSTM audio classification for statements vs. questions...")
    print(f"Using device: {device}")
    print("========================================================================\n")
    
    # Check if processed data exists
    processed_dir = "./processed_data"
    spec_file = os.path.join(processed_dir, "spectrograms.npy")
    features_file = os.path.join(processed_dir, "features.npy")
    labels_file = os.path.join(processed_dir, "labels.npy")
    
    if os.path.exists(spec_file) and os.path.exists(features_file) and os.path.exists(labels_file):
        print("\nFound preprocessed data. To use it directly for training, run: python train_model.py")
        print("To reprocess the data, run: python preprocess_data.py --force\n")
        
        user_input = input("Do you want to continue processing from scratch? (y/n): ")
        if user_input.lower() != 'y':
            print("\nExiting. Use the specialized scripts for separate preprocessing and training.")
            return
    
    # Original processing code
    # Get audio files
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
    print(f"Found {len(audio_files)} audio files.")
    
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
    
    print(f"Processed {len(spectrograms)} valid audio files")
    print(f"Feature vector shape: {feature_vectors.shape}")
    print(f"Label distribution: {np.sum(labels)} questions, {len(labels) - np.sum(labels)} statements")
    
    # Scale feature vectors
    scaler = StandardScaler()
    feature_vectors_scaled = scaler.fit_transform(feature_vectors)
    
    # Add channel dimension to spectrograms for CNN
    spectrograms = spectrograms.reshape(spectrograms.shape[0], 1, spectrograms.shape[1], spectrograms.shape[2])
    
    # Split data
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
    batch_size = 32 if torch.cuda.is_available() else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4 if torch.cuda.is_available() else 0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4 if torch.cuda.is_available() else 0)
    
    # Initialize model
    model = CNNLSTMClassifier(n_features=X_feat_train.shape[1], n_mels=spectrograms.shape[2])
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train model
    print("Starting model training...")
    train_losses, test_losses, accuracies = train_model(
        train_loader, test_loader, model, criterion, optimizer, device, num_epochs=50
    )
    
    # Move model back to CPU for saving (to make it compatible with CPU inference)
    model = model.to('cpu')
    
    # Save model and scaler
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
    print("Training complete!")

if __name__ == "__main__":
    main() 