# Audio Classification: Question vs. Statement Detection

This project implements a CNN+LSTM neural network model to classify audio clips as either questions or statements based on acoustic features. The system is structured in three stages:

1. **Feature Extraction**: Process audio files and extract acoustic features
2. **Model Training**: Train a CNN+LSTM model on the extracted features
3. **Demo Application**: Interactive GUI for real-time classification

## Features

- **Audio Feature Extraction**:
  - Mel Spectrograms: Represent the audio's frequency content with mel-scale filtering
  - MFCCs (Mel-frequency cepstral coefficients): Compact representation of audio spectral envelope
  - Prosodic Features: Pitch (F0), energy (RMS), and zero-crossing rate

- **Model Architecture**:
  - Hybrid CNN+LSTM neural network
  - CNN layers extract spatial features from the audio representations
  - LSTM layers capture temporal dynamics across the audio sequence
  - Combined features for final classification

- **Interactive Demo**:
  - Record live audio or load audio files
  - Visualize extracted features
  - Real-time classification with confidence scores

## Dataset

The system works with WAV audio files from the Switchboard Dialog Act (SWDA) corpus. The file naming convention is:
`sw[conversation_id][speaker]_[start_time]_[end_time].wav`

## Prerequisites

- Python 3.8+
- Dependencies:
  - PyTorch (for deep learning)
  - librosa (for audio processing)
  - numpy, pandas, scikit-learn (for data handling and evaluation)
  - matplotlib (for visualization)
  - PyQt5 (for the GUI application)
  - sounddevice, soundfile (for audio recording and playback)

Install dependencies:
```
pip install librosa torch numpy pandas scikit-learn matplotlib PyQt5 sounddevice soundfile tqdm
```

## Project Structure

- `feature_extractor.py`: Extracts and saves audio features
- `model_trainer.py`: Trains the CNN+LSTM model on extracted features
- `demo_app.py`: GUI application for interactive classification
- `main.py`: Central script to run any or all stages
- `processed_data/`: Directory for storing extracted features
- `models/`: Directory for storing trained models
- `swda_audio/`: Directory for storing all the audio file for the extractor

## Usage

The project can be run in three separate stages or all at once:

### 1. Feature Extraction

```
python main.py extract
```

This will:
- Process all audio files in the `swda_audio` directory
- Extract mel spectrograms, MFCCs, and prosodic features
- Save the features to `processed_data/processed_features.pkl`

### 2. Model Training

```
python main.py train
```

This will:
- Load the extracted features
- Train a CNN+LSTM model
- Save the best model to `models/best_model.pth`
- Generate evaluation metrics and visualizations

### 3. Demo Application

```
python main.py demo
```

This will launch an interactive GUI where you can:
- Record audio or load audio files
- View audio waveform and extracted features
- Classify the audio as a question or statement
- See confidence scores for the classification

### Run All Stages

```
python main.py all
```

This will run all three stages in sequence.

## How It Works

### Feature Extraction

The feature extraction stage:
- Loads each audio file and resamples to a consistent rate
- Extracts mel spectrograms, MFCCs, and prosodic features
- Saves processed features to disk for training

### Model Training

The model training stage:
- Loads preprocessed features
- Splits data into training and testing sets
- Uses a hybrid CNN+LSTM architecture
- Trains with Adam optimizer and cross-entropy loss
- Monitors accuracy on validation set
- Saves the best performing model

### Demo Application

The demo application:
- Provides an interactive GUI for audio classification
- Supports recording audio or loading audio files
- Visualizes extracted features
- Displays classification results with confidence scores

## Model Customization

You can adjust the following parameters in the respective scripts:
- Audio settings in `feature_extractor.py`
  - `SAMPLE_RATE`: Audio sample rate (default: 16000 Hz)
  - `MAX_AUDIO_LENGTH`: Maximum audio clip length in seconds (default: 5s)
  - `NUM_MELS`: Number of mel bands for spectrogram (default: 128)
  - `NUM_MFCC`: Number of MFCC coefficients (default: 40)
- Training parameters in `model_trainer.py`
  - Neural network architecture parameters in the `CNNLSTMModel` class
  - Training hyperparameters in the `train_model` function 