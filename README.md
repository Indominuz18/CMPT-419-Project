# Audio Question vs Statement Classification

A deep learning system that distinguishes between question and statement utterances using multi-modal acoustic features.

## Project Structure

```
project/
├── main.py                  # Main entry point for running all pipeline stages
├── feature_extractor.py     # Audio feature extraction functionality
├── model_trainer.py         # Neural network model training code
├── demo_app.py              # PyQt5 GUI application for visualization and classification
├── visualization.py # Visualization tools for project presentation
├── swda_audio/              # Directory for audio samples (not included)
├── processed_data/          # Directory for storing extracted features
└── models/                  # Directory for storing trained models
```

## Features

- **Advanced Acoustic Feature Extraction**: Extracts mel spectrograms, MFCCs, and prosodic features (pitch contours, energy, rhythm, etc.)
- **Multi-modal Deep Learning**: Uses a sophisticated neural network architecture with specialized pathways for each feature type
- **Interactive Demo Application**: PyQt5-based GUI for visualization and classification of audio files
- **Data Augmentation**: Implements time stretching, pitch shifting, and noise addition techniques
- **Feature Importance Visualization**: Tools for analyzing which acoustic features contribute most to classification

## Self-evaluation

This project successfully implemented all the core components proposed:

- ✅ Feature extraction pipeline for audio processing
- ✅ Neural network architecture for multi-modal classification
- ✅ GUI application for visualization and classification
- ✅ Comprehensive visualization tools

Enhancements beyond the original proposal:
- Added more sophisticated prosodic feature extraction
- Implemented attention mechanisms in the model architecture
- Created more detailed feature visualizations for interpretability
- Added data augmentation techniques specifically for question samples to address class imbalance


## Dependencies

The project requires the following dependencies:

```
python >= 3.8
torch >= 1.7.0
librosa >= 0.8.0
numpy >= 1.19.0
matplotlib >= 3.3.0
PyQt5 >= 5.15.0
scikit-learn >= 0.24.0
soundfile >= 0.10.3
tqdm >= 4.50.0
pandas >= 1.1.0
seaborn >= 0.11.0
```

## Installation & Usage

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create the necessary directories:
   ```
   mkdir -p swda_audio processed_data models
   ```
4. Add audio samples to the `swda_audio` directory (WAV files)
5. Run the full pipeline:
   ```
   python main.py all
   ```
   
   Or run individual stages:
   ```
   python main.py extract  # Extract features
   python main.py train    # Train model
   python main.py demo     # Run demo application
   ```

## Demo Application Usage

1. Launch the demo: `python demo_app.py`
2. Load audio samples using the "Load Sample" or "Load Audio File" buttons
3. Features will be automatically extracted and visualized
4. Classify audio using the "Classify" button

## Notes for TAs

- The model is designed to work with any WAV audio samples, not just those from the SWDA corpus
- For optimal performance, replace the `determine_label` function with your actual labeling logic using the SWDA corpus annotations
- The demo mode will work even without training your own model, as a pre-trained model is included

## Project Contributions

- **Enhanced Prosodic Feature Extraction**: Implemented sophisticated analysis of pitch contours, energy patterns, and spectral features specifically optimized for question-statement discrimination
- **Multi-Modal Deep Learning Architecture**: Developed a neural network that processes three feature streams (spectrograms, MFCCs, prosody) with specialized processing pathways and attention mechanisms
- **Advanced Data Augmentation**: Created targeted augmentation techniques to improve model robustness and address class imbalance
- **Interactive Classification Interface**: Built a PyQt5-based GUI application for visualization and classification of audio samples
- **Feature Importance Visualization**: Developed visualization tools that provide interpretable insights into the acoustic features that differentiate questions from statements

## Visualization for Presentations

The project includes a dedicated visualization script for generating high-quality graphics suitable for presentations and reports:

```
python visualization.py
```

This will generate the following visualizations in the `visuals` directory:

- **Model Architecture** (`model_architecture.png`): Visual representation of the neural network architecture
- **Training Performance** (`training_curves.png`): Training/validation loss and F1 score plots
- **Model Performance** (`model_performance.png`): Confusion matrix and performance metrics
- **Feature Importance** (`feature_importance.png`): Analysis of feature contributions to classification
- **Sample Analyses** (`sample_X_analysis.png`): Detailed visualizations of audio samples with features and predictions
- **Prosody Analysis** (`prosody_analysis.png`): Comparison of prosodic features between questions and statements

These visualizations use a consistent color scheme (purple for questions, orange for statements) and are generated at 300 DPI resolution, making them suitable for presentations, posters, and publications.

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
  - soundfile (for audio loading and playback)

Install dependencies:
```
pip install librosa torch numpy pandas scikit-learn matplotlib PyQt5 soundfile tqdm
```

## Project Structure

- `feature_extractor.py`: Extracts and saves audio features
- `model_trainer.py`: Trains the CNN+LSTM model on extracted features
- `demo_app.py`: GUI application for visualization and classification
- `main.py`: Central script to run any or all stages
- `processed_data/`: Directory for storing extracted features
- `models/`: Directory for storing trained models

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
- Load audio samples using the "Load Sample" or "Load Audio File" buttons
- Features will be automatically extracted and visualized
- Classify audio using the "Classify" button

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
- Labels audio clips based on heuristics or annotations
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
- Supports loading audio samples
- Visualizes extracted features
- Displays classification results with confidence scores

## Note on Labeling

In a real-world scenario, you would need ground truth labels for questions and statements. The current implementation uses a simple heuristic based on pitch rise at the end of the utterance to create labels. For optimal performance, replace the `determine_label` function with your actual labeling logic using the SWDA corpus annotations.

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