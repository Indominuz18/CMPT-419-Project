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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define paths
AUDIO_DIR = "../swda_audio"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to extract audio features
def extract_features(audio_path, n_mfcc=13, n_mels=40):
    """Extract audio features from an audio file."""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        
        # Additional features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Extract pitch (F0) contour which is important for questions
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        # Compute statistics for each feature
        features = []
        
        # MFCC statistics (mean, std, min, max)
        for e in [mfccs, mel_spectrogram, chroma, spectral_contrast]:
            features.extend([np.mean(e), np.std(e), np.min(e), np.max(e)])
        
        # For F0, we're particularly interested in the trend (rising F0 is often a question)
        # First, remove any NaN or infinity values
        f0_clean = f0[~np.isnan(f0) & ~np.isinf(f0)]
        if len(f0_clean) > 0:
            # Compute trend (positive slope often indicates question)
            if len(f0_clean) > 1:
                f0_trend = np.polyfit(np.arange(len(f0_clean)), f0_clean, 1)[0]
            else:
                f0_trend = 0
                
            features.extend([np.mean(f0_clean), np.std(f0_clean), f0_trend])
        else:
            features.extend([0, 0, 0])  # Default values if no valid F0
            
        # Add energy and its variation
        energy = np.sum(y**2) / len(y)
        features.append(energy)
            
        return np.array(features)
    
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

# Define Dataset class
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Define model architecture
class AudioClassifier(nn.Module):
    def __init__(self, input_size):
        super(AudioClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

# Main function
def main():
    print("Starting audio classification for statements vs. questions...")
    
    # Step 1: Prepare data with labels
    # For this example, we'll create synthetic labels (0=statement, 1=question)
    # In a real scenario, you would need actual labels for your audio files
    
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
    
    # For demonstration purposes, we'll use a heuristic for labeling:
    # Audio files with rising intonation at the end are more likely to be questions
    # This is just a placeholder - in reality, you'd use actual labels
    labels = []
    features_list = []
    
    print(f"Processing {len(audio_files)} audio files...")
    for audio_file in tqdm(audio_files):
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        features = extract_features(audio_path)
        
        if features is not None:
            features_list.append(features)
            
            # Placeholder labeling logic - you would replace this with real labels
            # For this demo, we'll consider files with "B_" as likely questions (60%)
            # and files with "A_" as likely statements (80%)
            if "_B_" in audio_file:
                label = 1 if np.random.random() < 0.6 else 0  # 60% chance of being a question
            else:
                label = 0 if np.random.random() < 0.8 else 1  # 80% chance of being a statement
            
            labels.append(label)
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels)
    
    print(f"Extracted features for {len(X)} audio files")
    print(f"Label distribution: {np.sum(y)} questions, {len(y) - np.sum(y)} statements")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    # Create datasets and dataloaders
    train_dataset = AudioDataset(X_train_tensor, y_train_tensor)
    test_dataset = AudioDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_size = X_train.shape[1]
    model = AudioClassifier(input_size)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 100
    train_losses = []
    test_losses = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_losses.append(test_loss)
        accuracy = correct / total
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "statement_question_model.pth"))
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curve.png"))
    
    # Evaluate final model
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    # Create a function to classify new audio
    def classify_audio(audio_path, model, scaler):
        features = extract_features(audio_path)
        if features is None:
            return None
            
        features_scaled = scaler.transform(features.reshape(1, -1))
        features_tensor = torch.FloatTensor(features_scaled)
        
        model.eval()
        with torch.no_grad():
            output = model(features_tensor)
            prediction = "Question" if output.item() >= 0.5 else "Statement"
            confidence = max(output.item(), 1 - output.item())
            
        return prediction, confidence
    
    # Save the scaler for future use
    import pickle
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"Model saved to {os.path.join(OUTPUT_DIR, 'statement_question_model.pth')}")
    print("Training complete!")
    
    # Test the model on a few examples
    print("\nTesting on a few examples:")
    for i in range(min(5, len(audio_files))):
        audio_path = os.path.join(AUDIO_DIR, audio_files[i])
        prediction, confidence = classify_audio(audio_path, model, scaler)
        print(f"File: {audio_files[i]}")
        print(f"Prediction: {prediction} (Confidence: {confidence:.2f})")
        print()

if __name__ == "__main__":
    main() 