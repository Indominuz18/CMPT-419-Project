# Audio Classification Workflow Guide

This guide explains how to use the separated data processing and model training workflow for the audio classification project.

## Why Separate Processing and Training?

The original script (`cnn_lstm_classifier.py`) performs both feature extraction and model training in a single step. However, this means you need to reprocess all audio files every time you want to experiment with different training parameters, which can be time-consuming.

By separating these steps, you can:
1. Process your data once
2. Run multiple training experiments with different parameters
3. Save time when iterating on the model architecture or hyperparameters

## Workflow Steps

### Step 1: Preprocess Data

Run the preprocessing script to extract features from all audio files and save them to disk:

```bash
python preprocess_data.py
```

This will:
- Create a `processed_data` directory
- Extract spectrograms and features from all audio files
- Save the data as NumPy arrays
- Generate labels (based on the heuristic used in the original script)

If you want to force reprocessing (e.g., after changing the feature extraction code):

```bash
python preprocess_data.py --force
```

### Step 2: Train the Model

Once you have preprocessed data, you can train the model:

```bash
python train_model.py
```

The training script provides several command-line options:
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 32 with GPU, 16 without)
- `--lr`: Learning rate (default: 0.001)

For example, to train for 100 epochs with a larger batch size:

```bash
python train_model.py --epochs 100 --batch-size 64
```

The trained model and scaler will be saved to the `output` directory, just like in the original workflow.

### Step 3: Classify Audio

The classification script remains the same:

```bash
python classify_audio_cnn_lstm.py /path/to/your/audio.wav
```

Or use the GUI demo:

```bash
python demo_classifier.py
```

## Alternative: Original Workflow

If you prefer the original all-in-one approach, you can still use:

```bash
python cnn_lstm_classifier.py
```

The script will now check if preprocessed data exists and give you the option to use the separated workflow instead.

## Tips for Experimentation

When experimenting with model architecture or training parameters:

1. First extract features using `preprocess_data.py`
2. Try different training configurations using `train_model.py`
3. Keep the best performing model in the `output` directory
4. Test the model on new audio files using `classify_audio_cnn_lstm.py`

This approach allows you to focus on model optimization without repeatedly processing the same data. 