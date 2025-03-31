import os
import sys
import torch
import pickle
import numpy as np
from statement_question_classifier import AudioClassifier, extract_features

def classify_audio_file(audio_path, model_path, scaler_path):
    """
    Classify an audio file as either a statement or a question.
    
    Args:
        audio_path: Path to the audio file
        model_path: Path to the trained model
        scaler_path: Path to the saved scaler
    
    Returns:
        Prediction (string) and confidence (float)
    """
    # Check if files exist
    if not os.path.exists(audio_path):
        print(f"Error: Audio file {audio_path} not found.")
        return None
        
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return None
        
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file {scaler_path} not found.")
        return None
    
    # Extract features
    features = extract_features(audio_path)
    if features is None:
        print(f"Error: Could not extract features from {audio_path}")
        return None
    
    # Load scaler and scale features
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_tensor = torch.FloatTensor(features_scaled)
    
    # Load model
    input_size = features_scaled.shape[1]
    model = AudioClassifier(input_size)
    model.load_state_dict(torch.load(model_path))
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(features_tensor)
        prediction = "Question" if output.item() >= 0.5 else "Statement"
        confidence = max(output.item(), 1 - output.item())
    
    return prediction, confidence

def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_audio.py <audio_file_path>")
        return
    
    audio_path = sys.argv[1]
    model_path = "./output/statement_question_model.pth"
    scaler_path = "./output/scaler.pkl"
    
    result = classify_audio_file(audio_path, model_path, scaler_path)
    
    if result:
        prediction, confidence = result
        print(f"Audio file: {audio_path}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main() 