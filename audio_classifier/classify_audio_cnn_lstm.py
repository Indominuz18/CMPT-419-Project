import os
import sys
import numpy as np
import torch
import pickle
import warnings
from cnn_lstm_classifier import CNNLSTMClassifier, extract_features

# Suppress warnings
warnings.filterwarnings("ignore")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def classify_audio_file(audio_path, model_path, scaler_path):
    """
    Classify an audio file as either a statement or a question using CNN+LSTM model.
    
    Args:
        audio_path: Path to the audio file
        model_path: Path to the trained CNN+LSTM model
        scaler_path: Path to the saved scaler
    
    Returns:
        Prediction (string) and confidence (float)
    """
    print("\n" + "="*70)
    print(f"CLASSIFYING: {os.path.basename(audio_path)}")
    print("="*70)
    
    # Check if files exist
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file {audio_path} not found.")
        return None
        
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file {model_path} not found.")
        return None
        
    if not os.path.exists(scaler_path):
        print(f"❌ Error: Scaler file {scaler_path} not found.")
        return None
    
    print("\n📊 Processing audio file...")
    # Extract spectrogram and features
    spectrogram, feature_vector = extract_features(audio_path)
    if spectrogram is None or feature_vector is None:
        print(f"❌ Error: Could not extract features from {audio_path}")
        return None
    
    # Verify feature shape
    if len(feature_vector) != 11:
        print(f"❌ Error: Feature vector has unexpected length: {len(feature_vector)}")
        return None
    
    print("✅ Features extracted successfully")
    
    # Load scaler and scale features
    print("\n🔍 Applying feature scaling...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Scale features
    feature_vector_scaled = scaler.transform(feature_vector.reshape(1, -1))
    
    # Add channel dimension to spectrogram for CNN
    spectrogram = spectrogram.reshape(1, 1, spectrogram.shape[0], spectrogram.shape[1])
    
    # Convert to PyTorch tensors
    spectrogram_tensor = torch.FloatTensor(spectrogram)
    feature_vector_tensor = torch.FloatTensor(feature_vector_scaled)
    
    # Move tensors to the appropriate device
    spectrogram_tensor = spectrogram_tensor.to(device)
    feature_vector_tensor = feature_vector_tensor.to(device)
    
    # Load model
    print("🧠 Loading model...")
    model = CNNLSTMClassifier(n_features=feature_vector_scaled.shape[1], n_mels=spectrogram.shape[2])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Make prediction
    print("🔮 Making prediction...")
    model.eval()
    with torch.no_grad():
        output = model(spectrogram_tensor, feature_vector_tensor)
        prediction = "Question" if output.item() >= 0.5 else "Statement"
        confidence = max(output.item(), 1 - output.item())
    
    # Print feature information for analysis
    print("\n" + "-"*70)
    print("ACOUSTIC FEATURES ANALYSIS:")
    print("-"*70)
    
    # Format features as a table
    features = [
        ("Pitch trend (rising/falling)", feature_vector[8], "↗ Rising" if feature_vector[8] > 0 else "↘ Falling"),
        ("Average pitch", feature_vector[6], "Hz"),
        ("Pitch variation", feature_vector[7], "Standard deviation"),
        ("Energy (loudness)", feature_vector[9], "RMS"),
        ("Zero-crossing rate", feature_vector[10], "Rate")
    ]
    
    # Print features
    for name, value, unit in features:
        print(f"{name:30s} | {value:8.3f} | {unit}")
    
    # Print prediction result
    print("\n" + "-"*70)
    print(f"PREDICTION: {prediction.upper()} ({confidence*100:.1f}% confidence)")
    print("-"*70)
    
    # Interpretation
    print("\nLINGUISTIC INTERPRETATION:")
    if prediction == "Question":
        print("• This utterance exhibits acoustic patterns typical of questions")
        print("• Rising intonation (↗) at the end is a key indicator")
        if feature_vector[8] > 0:
            print("• The rising pitch trend confirms question-like prosody")
        else:
            print("• Despite falling pitch trend, other features suggest a question")
    else:
        print("• This utterance exhibits acoustic patterns typical of statements")
        print("• Falling or flat intonation (↘) is a key indicator")
        if feature_vector[8] <= 0:
            print("• The falling pitch trend confirms statement-like prosody")
        else:
            print("• Despite rising pitch trend, other features suggest a statement")
    
    print("\n" + "="*70 + "\n")
    
    return prediction, confidence

def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_audio_cnn_lstm.py <audio_file_path>")
        return
    
    audio_path = sys.argv[1]
    model_path = "./output/cnn_lstm_model.pth"
    scaler_path = "./output/cnn_lstm_scaler.pkl"
    
    print(f"Processing audio file: {audio_path}")
    print(f"Using GPU: {'Yes' if torch.cuda.is_available() else 'No'}")
    
    result = classify_audio_file(audio_path, model_path, scaler_path)
    
    if result:
        prediction, confidence = result
        print(f"\nAudio file: {audio_path}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}")
        
        # Provide linguistic interpretation
        if prediction == "Question":
            print("\nAcoustic analysis: This utterance likely has rising intonation at the end,")
            print("which is characteristic of questions in English and many other languages.")
        else:
            print("\nAcoustic analysis: This utterance likely has falling or flat intonation,")
            print("which is characteristic of statements or declarations.")

if __name__ == "__main__":
    main() 