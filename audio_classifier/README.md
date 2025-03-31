# Audio Statement vs. Question Classifier

This project provides models to classify audio clips as either statements or questions based on acoustic features.

## Project Overview

The repository includes two approaches:
1. A traditional ML approach using extracted audio features (statement_question_classifier.py)
2. A deep learning approach using CNN+LSTM architecture with spectrograms (cnn_lstm_classifier.py)

The CNN+LSTM model is more powerful as it:
- Uses MEL spectrograms to capture frequency information over time
- Extracts essential acoustic features (MFCCs, pitch, energy)
- Applies CNN layers to extract spatial features from spectrograms
- Uses bidirectional LSTM to capture temporal patterns in the audio
- Focuses on the most reliable prosodic cues for question detection
- Leverages GPU acceleration for faster training and inference

## Requirements

Install required dependencies:
```
pip install -r requirements.txt
```

## Hardware Requirements

- For optimal performance: NVIDIA GPU with CUDA support
- The model will automatically detect and use GPU if available
- Falls back to CPU if no GPU is detected

## Dataset

This project uses the Switchboard Dialog Act (SWDA) corpus audio files. The files follow naming conventions like `sw04917A_125466_13040975.wav` where:
- Speaker ID (A or B) indicates the person speaking
- Numbers indicate specific segments in the conversation

## Training Models

### Training the CNN+LSTM Model

```
python cnn_lstm_classifier.py
```

This will:
- Extract spectrograms and acoustic features from audio files
- Train a CNN+LSTM model using GPU acceleration (if available)
- Save the trained model and scaler to the `output` directory
- Generate training plots showing loss and accuracy

## Classifying New Audio

### Using the CNN+LSTM Model

```
python classify_audio_cnn_lstm.py /path/to/your/audio.wav
```

## Key Features Used

The model has been simplified to focus on the most reliable features for statement/question classification:

1. **MEL Spectrograms**
   - Time-frequency representation processed by CNN
   - Captures the overall acoustic pattern of speech

2. **MFCC Features**
   - Mel-Frequency Cepstral Coefficients (mean and std)
   - Captures the spectral envelope of speech

3. **Pitch Features**
   - Final pitch trend (rising/falling intonation)
   - Mean and variation of fundamental frequency
   - Critical for question detection (rising pitch = question)

4. **Spectral Features**
   - Spectral centroid (brightness of sound)
   - Spectral bandwidth (width of the spectrum)

5. **Energy and Zero-Crossing Rate**
   - Overall loudness of the audio
   - Voice quality measurements

## Model Architecture

### CNN+LSTM Model

1. **CNN Layers**:
   - 4 convolutional blocks with batch normalization
   - Extracts spatial patterns from spectrograms
   - Progressively increases feature depth (32→64→128→256)

2. **LSTM Layers**:
   - Bidirectional LSTM with 2 layers and 256 hidden units
   - Captures temporal patterns and sequence information

3. **Classification Layers**:
   - Combines LSTM features with extracted acoustic features
   - Fully connected layers with dropout for regularization
   - Final sigmoid activation for binary classification

## Linguistic Significance

Questions and statements differ primarily in their prosodic features:

1. **Terminal Contour**: Questions typically end with rising pitch (↗), while statements end with falling pitch (↘)
2. **Pitch Range**: Questions often have wider pitch variation
3. **Energy Patterns**: Different stress patterns in interrogative vs. declarative sentences

The model is specifically designed to detect these key acoustic differences for reliable classification. 