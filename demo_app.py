import os
import sys
import pickle
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sounddevice as sd
import soundfile as sf
import tempfile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QFileDialog, QLabel, QWidget, QFrame, 
                             QSpacerItem, QSizePolicy, QProgressBar, QGroupBox, QComboBox, QMessageBox,
                             QListWidget, QDialog, QListWidgetItem)
from PyQt5.QtGui import QFont, QPixmap, QColor, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

# Import the feature extraction functions
from feature_extractor import extract_features
# Import torch modules for model definition
import torch.nn as nn

# Constants
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 5
NUM_MELS = 128
NUM_MFCC = 40
RECORDING_DURATION = 5  # seconds
MODEL_PATH = 'models/best_model.pth'
MODEL_INFO_PATH = 'models/model_info.pkl'

# Simple model class for loading the saved model
class SimplifiedModel(nn.Module):
    """A simplified model that matches the structure of the saved model for demo purposes only"""
    def __init__(self, device=None):
        super(SimplifiedModel, self).__init__()
        self.device = device if device is not None else torch.device('cpu')
        
        # CNN layers (simplified versions of the training model)
        self.cnn_mel = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.cnn_mfcc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.cnn_prosody = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # LSTM layers with fixed dimensions
        self.lstm_mel = nn.LSTM(3968, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm_mfcc = nn.LSTM(3968, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm_prosody = nn.LSTM(4000, 64, num_layers=2, batch_first=True, bidirectional=True)
        
        # Save dimensions for reference
        self.lstm_mel_input_dim = 3968
        self.lstm_mfcc_input_dim = 3968
        self.lstm_prosody_input_dim = 4000
        
        # Attention mechanisms
        self.attention_mel = nn.Linear(256, 256)  # Simplified version
        self.attention_mfcc = nn.Linear(128, 128)
        self.attention_prosody = nn.Linear(128, 128)
        
        # Layer normalization
        self.layer_norm_mel = nn.LayerNorm(256)
        self.layer_norm_mfcc = nn.LayerNorm(128)
        self.layer_norm_prosody = nn.LayerNorm(128)
        
        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        
        # Explicitly move all components to the specified device
        self.to(self.device)
    
    def to(self, device):
        """Override to method to ensure all components move to the right device"""
        self.device = device
        return super(SimplifiedModel, self).to(device)

    def forward(self, mel_spec, mfcc, prosody):
        """Forward pass for inference"""
        batch_size = mel_spec.size(0)
        
        # Process mel spectrogram
        mel_out = self.cnn_mel(mel_spec)
        mel_out = mel_out.permute(0, 2, 1, 3)  # [batch, seq_len, channels, features]
        mel_out = mel_out.reshape(batch_size, mel_out.size(1), -1)  # Flatten for LSTM
        
        # Process MFCC
        mfcc = mfcc.unsqueeze(1)  # Add channel dimension
        mfcc_out = self.cnn_mfcc(mfcc)
        mfcc_out = mfcc_out.permute(0, 2, 1, 3)
        mfcc_out = mfcc_out.reshape(batch_size, mfcc_out.size(1), -1)
        
        # Process prosody
        prosody = prosody.unsqueeze(1)  # Add channel dimension
        prosody_out = self.cnn_prosody(prosody)
        prosody_out = prosody_out.permute(0, 2, 1, 3)
        prosody_out = prosody_out.reshape(batch_size, prosody_out.size(1), -1)
        
        # Pass through LSTM layers
        mel_out, _ = self.lstm_mel(mel_out)
        mfcc_out, _ = self.lstm_mfcc(mfcc_out)
        prosody_out, _ = self.lstm_prosody(prosody_out)
        
        # Apply simplified attention (just applying linear layer)
        mel_out = self.attention_mel(mel_out)
        mfcc_out = self.attention_mfcc(mfcc_out)
        prosody_out = self.attention_prosody(prosody_out)
        
        # Apply layer normalization
        mel_out = self.layer_norm_mel(mel_out)
        mfcc_out = self.layer_norm_mfcc(mfcc_out)
        prosody_out = self.layer_norm_prosody(prosody_out)
        
        # Get sequence-level representations by mean pooling
        mel_out = torch.mean(mel_out, dim=1)
        mfcc_out = torch.mean(mfcc_out, dim=1)
        prosody_out = torch.mean(prosody_out, dim=1)
        
        # Concatenate all features
        combined = torch.cat((mel_out, mfcc_out, prosody_out), dim=1)
        
        # Final classification
        output = self.fc(combined)
        return output

# Recording thread
class AudioRecorder(QThread):
    """Thread for recording audio without blocking the GUI"""
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    
    def __init__(self, duration=RECORDING_DURATION, sample_rate=SAMPLE_RATE, device=None):
        super().__init__()
        self.duration = duration
        self.sample_rate = sample_rate
        self.device = device
        
    def run(self):
        try:
            print(f"Recording {self.duration} seconds of audio...")
            # Try to record using sounddevice
            try:
                audio = sd.rec(int(self.duration * self.sample_rate), 
                              samplerate=self.sample_rate, 
                              channels=1,
                              device=self.device)
                sd.wait()  # Wait until recording is finished
                audio = audio.flatten()  # Flatten the array to 1D
                self.finished.emit(audio)
            except Exception as e:
                # If sounddevice fails, try a more direct approach
                print(f"Primary recording method failed: {e}")
                self.error.emit(f"Could not record audio: {e}\nPlease use the Load Sample or Load Audio File options instead.")
        except Exception as e:
            print(f"Recording error: {e}")
            self.error.emit(str(e))

class SampleFilesDialog(QDialog):
    """Dialog for selecting sample audio files"""
    def __init__(self, parent=None, sample_dir='swda_audio'):
        super().__init__(parent)
        self.setWindowTitle("Select Sample Audio File")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        # Store the sample directory
        self.sample_dir = sample_dir
        
        # Setup UI
        layout = QVBoxLayout()
        
        # Label
        label = QLabel("Select a sample audio file to load:")
        layout.addWidget(label)
        
        # List widget for files
        self.list_widget = QListWidget()
        self.populate_files()
        layout.addWidget(self.list_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Load button
        load_button = QPushButton("Load Selected")
        load_button.clicked.connect(self.accept)
        button_layout.addWidget(load_button)
        
        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def populate_files(self):
        """Populate the list with sample audio files"""
        if not os.path.exists(self.sample_dir):
            self.list_widget.addItem("Sample directory not found!")
            return
        
        files = [f for f in os.listdir(self.sample_dir) if f.endswith('.wav')]
        
        if not files:
            self.list_widget.addItem("No sample files found!")
            return
        
        for f in sorted(files):
            self.list_widget.addItem(f)
    
    def get_selected_file(self):
        """Get the selected file path"""
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return None
        
        filename = selected_items[0].text()
        return os.path.join(self.sample_dir, filename)

# Feature visualization
class FeatureVisualization(FigureCanvas):
    """Canvas for visualizing audio features"""
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(FeatureVisualization, self).__init__(self.fig)
        self.setMinimumSize(300, 200)
        
    def plot_waveform(self, audio, sample_rate):
        self.axes.clear()
        self.axes.set_title('Waveform')
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Amplitude')
        time = np.arange(0, len(audio)) / sample_rate
        self.axes.plot(time, audio)
        self.fig.tight_layout()
        self.draw()
    
    def plot_spectrogram(self, spectrogram):
        self.axes.clear()
        self.axes.set_title('Mel Spectrogram')
        self.axes.set_ylabel('Mel Bands')
        self.axes.set_xlabel('Time')
        
        # Check if spectrogram has multiple channels (3D)
        if len(spectrogram.shape) == 3:
            # Use the first channel for visualization
            spectrogram_2d = spectrogram[0]
            print(f"Using first channel of spectrogram with shape {spectrogram.shape}")
        else:
            spectrogram_2d = spectrogram
        
        img = self.axes.imshow(spectrogram_2d, aspect='auto', origin='lower', cmap='viridis')
        self.fig.colorbar(img, ax=self.axes)
        self.fig.tight_layout()
        self.draw()
    
    def plot_mfcc(self, mfcc):
        mfcc_display = mfcc[:NUM_MFCC]  # Display only the base MFCCs, not deltas
        self.axes.clear()
        self.axes.set_title('MFCC Features')
        self.axes.set_ylabel('MFCC Coefficients')
        self.axes.set_xlabel('Time')
        img = self.axes.imshow(mfcc_display, aspect='auto', origin='lower', cmap='cool')
        self.fig.colorbar(img, ax=self.axes)
        self.fig.tight_layout()
        self.draw()
        
    def plot_pitch(self, pitch):
        self.axes.clear()
        self.axes.set_title('Pitch (F0)')
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Frequency (normalized)')
        self.axes.plot(pitch)
        self.fig.tight_layout()
        self.draw()

class AudioClassifierApp(QMainWindow):
    """Main application window for audio classification demo"""
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Question vs. Statement Audio Classifier")
        self.setGeometry(100, 100, 1000, 700)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Setup UI first
        self.setup_ui()
        
        # Then load model after UI is set up
        self.load_model()
        
        # Initialize audio and features
        self.audio = None
        self.features = None
        self.is_recording = False
        self.recorder = None
        
    def load_model(self):
        """Load the trained model"""
        try:
            # Create a status label if it doesn't exist yet
            if not hasattr(self, 'status_label'):
                self.status_label = QLabel("Loading model...")
                print("Created status label during model loading")
            
            self.status_label.setText("Loading model...")
            
            # First try to load the demo model specifically created for the demo
            model_path = 'models/demo_model.pth'
            
            # If demo model doesn't exist, fall back to best model
            if not os.path.exists(model_path):
                model_path = 'models/best_model.pth'
                
            # If neither exists, show error
            if not os.path.exists(model_path):
                self.status_label.setText("Error: Model file not found!")
                return False
                
            # Load the model info first to get dimensions
            try:
                with open('models/model_info.pkl', 'rb') as f:
                    self.model_info = pickle.load(f)
                print(f"Loaded model info: {self.model_info}")
            except Exception as e:
                print(f"Warning: Could not load model_info.pkl: {e}")
                self.model_info = None
            
            # Create the model with fixed structure for demo purposes
            self.model = SimplifiedModel(device=self.device)
            print(f"Created model on device: {self.device}")
            
            # Set model to eval mode
            self.model.eval()
            
            # Load checkpoint - make sure it's on the same device
            self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            print(f"Loaded checkpoint from {model_path} to device {self.device}")
            
            # Get LSTM dimensions from the checkpoint
            lstm_dimensions = self.checkpoint.get('lstm_dimensions', {})
            if lstm_dimensions:
                print(f"Using LSTM dimensions from checkpoint: {lstm_dimensions}")
            
            # Load model state dict
            try:
                # First try direct loading of weights
                self.model.load_state_dict(self.checkpoint['model_state_dict'], strict=False)
                print("Model weights loaded with strict=False")
                
                # Ensure model is on the correct device AFTER loading weights
                self.model = self.model.to(self.device)
                print(f"Model explicitly moved to {self.device}")
            except Exception as e:
                print(f"Error loading model weights: {e}")
                return False
            
            print(f"Model initialized with LSTM dimensions from checkpoint: mel={self.model.lstm_mel_input_dim}, mfcc={self.model.lstm_mfcc_input_dim}, prosody={self.model.lstm_prosody_input_dim}")
            
            # Update status
            self.status_label.setText(f"Model loaded successfully! (F1 Score: {self.checkpoint.get('f1_score', 'N/A'):.4f})")
            return True
            
        except Exception as e:
            import traceback
            self.status_label.setText(f"Error loading model: {str(e)}")
            print(f"Error loading model: {e}")
            traceback.print_exc()
            return False
        
    def setup_ui(self):
        """Set up the user interface"""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Title
        title_label = QLabel("Question vs. Statement Audio Classifier")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('Arial', 16, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # Status label - create this early
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        font = QFont('Arial', 10)
        font.setItalic(True)
        self.status_label.setFont(font)
        main_layout.addWidget(self.status_label)
        
        # Audio device selection
        devices_group = QGroupBox("Audio Device Selection")
        devices_layout = QHBoxLayout()
        devices_group.setLayout(devices_layout)
        
        # Device selection label
        device_label = QLabel("Input Device:")
        devices_layout.addWidget(device_label)
        
        # Device dropdown
        self.device_combo = QComboBox()
        devices_layout.addWidget(self.device_combo)
        
        # Refresh devices button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.populate_audio_devices)
        devices_layout.addWidget(refresh_button)
        
        main_layout.addWidget(devices_group)
        
        # Controls section
        controls_group = QGroupBox("Audio Controls")
        controls_layout = QHBoxLayout()
        controls_group.setLayout(controls_layout)
        
        # Record button
        self.record_button = QPushButton("Record (5s)")
        self.record_button.setFont(QFont('Arial', 10))
        self.record_button.clicked.connect(self.toggle_recording)
        controls_layout.addWidget(self.record_button)
        
        # Load sample button
        load_sample_button = QPushButton("Load Sample")
        load_sample_button.setFont(QFont('Arial', 10))
        load_sample_button.clicked.connect(self.load_sample_audio)
        controls_layout.addWidget(load_sample_button)
        
        # Load audio button
        load_button = QPushButton("Load Audio File")
        load_button.setFont(QFont('Arial', 10))
        load_button.clicked.connect(self.load_audio)
        controls_layout.addWidget(load_button)
        
        # Classify button
        self.classify_button = QPushButton("Classify Audio")
        self.classify_button.setFont(QFont('Arial', 10))
        self.classify_button.clicked.connect(self.classify_audio)
        self.classify_button.setEnabled(False)
        controls_layout.addWidget(self.classify_button)
        
        # Play button
        self.play_button = QPushButton("Play Audio")
        self.play_button.setFont(QFont('Arial', 10))
        self.play_button.clicked.connect(self.play_audio)
        self.play_button.setEnabled(False)
        controls_layout.addWidget(self.play_button)
        
        main_layout.addWidget(controls_group)
        
        # Recording progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Features visualization section
        features_layout = QHBoxLayout()
        
        # Waveform
        waveform_group = QGroupBox("Waveform")
        waveform_layout = QVBoxLayout()
        self.waveform_canvas = FeatureVisualization(width=5, height=3)
        waveform_layout.addWidget(self.waveform_canvas)
        waveform_group.setLayout(waveform_layout)
        features_layout.addWidget(waveform_group)
        
        # Spectrogram
        spectrogram_group = QGroupBox("Mel Spectrogram")
        spectrogram_layout = QVBoxLayout()
        self.spectrogram_canvas = FeatureVisualization(width=5, height=3)
        spectrogram_layout.addWidget(self.spectrogram_canvas)
        spectrogram_group.setLayout(spectrogram_layout)
        features_layout.addWidget(spectrogram_group)
        
        main_layout.addLayout(features_layout)
        
        # Second row of features
        features_layout2 = QHBoxLayout()
        
        # MFCC
        mfcc_group = QGroupBox("MFCC Features")
        mfcc_layout = QVBoxLayout()
        self.mfcc_canvas = FeatureVisualization(width=5, height=3)
        mfcc_layout.addWidget(self.mfcc_canvas)
        mfcc_group.setLayout(mfcc_layout)
        features_layout2.addWidget(mfcc_group)
        
        # Pitch
        pitch_group = QGroupBox("Pitch (F0)")
        pitch_layout = QVBoxLayout()
        self.pitch_canvas = FeatureVisualization(width=5, height=3)
        pitch_layout.addWidget(self.pitch_canvas)
        pitch_group.setLayout(pitch_layout)
        features_layout2.addWidget(pitch_group)
        
        main_layout.addLayout(features_layout2)
        
        # Classification result section
        result_group = QGroupBox("Classification Result")
        result_layout = QVBoxLayout()
        
        # Result label
        self.result_label = QLabel("Record or load audio to classify")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont('Arial', 14, QFont.Bold))
        result_layout.addWidget(self.result_label)
        
        # Confidence bars
        confidence_layout = QHBoxLayout()
        
        # Statement confidence
        statement_layout = QVBoxLayout()
        statement_label = QLabel("Statement")
        statement_label.setAlignment(Qt.AlignCenter)
        statement_layout.addWidget(statement_label)
        
        self.statement_progress = QProgressBar()
        self.statement_progress.setRange(0, 100)
        self.statement_progress.setValue(0)
        statement_layout.addWidget(self.statement_progress)
        confidence_layout.addLayout(statement_layout)
        
        # Question confidence
        question_layout = QVBoxLayout()
        question_label = QLabel("Question")
        question_label.setAlignment(Qt.AlignCenter)
        question_layout.addWidget(question_label)
        
        self.question_progress = QProgressBar()
        self.question_progress.setRange(0, 100)
        self.question_progress.setValue(0)
        question_layout.addWidget(self.question_progress)
        confidence_layout.addLayout(question_layout)
        
        result_layout.addLayout(confidence_layout)
        result_group.setLayout(result_layout)
        main_layout.addWidget(result_group)
        
        # Now that all UI elements are created, populate the devices
        self.populate_audio_devices()
    
    def populate_audio_devices(self):
        """Populate the audio device dropdown with available input devices"""
        try:
            self.device_combo.clear()
            devices = sd.query_devices()
            input_devices = []
            
            # Add 'Default' option
            self.device_combo.addItem("Default", None)
            
            # Add each input device
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    name = f"{i}: {device['name']}"
                    self.device_combo.addItem(name, i)
                    input_devices.append(name)
            
            if not input_devices:
                self.status_label.setText("No input devices found! You can only load audio files.")
                self.record_button.setEnabled(False)
            else:
                self.status_label.setText(f"Found {len(input_devices)} input devices")
                self.record_button.setEnabled(True)
                
        except Exception as e:
            self.status_label.setText(f"Error loading audio devices: {e}")
            self.record_button.setEnabled(False)
    
    def toggle_recording(self):
        """Start or stop audio recording"""
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.record_button.setText("Stop Recording")
            self.classify_button.setEnabled(False)
            self.play_button.setEnabled(False)
            self.status_label.setText("Recording...")
            
            # Show and reset progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Get selected device
            device_idx = self.device_combo.currentData()
            
            # Start recording thread
            self.recorder = AudioRecorder(device=device_idx)
            self.recorder.finished.connect(self.process_recorded_audio)
            self.recorder.error.connect(self.handle_recording_error)
            self.recorder.start()
            
            # Update progress bar
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_progress)
            self.timer.start(50)  # Update every 50ms
        else:
            # Stop recording (if it's still going)
            if self.recorder and self.recorder.isRunning():
                sd.stop()
                self.recorder.wait()
            
            # Reset UI
            self.record_button.setText("Record (5s)")
            self.is_recording = False
            self.progress_bar.setVisible(False)
            if hasattr(self, 'timer') and self.timer.isActive():
                self.timer.stop()
    
    def update_progress(self):
        """Update the progress bar during recording"""
        if hasattr(self, 'recorder') and self.recorder.isRunning():
            current = self.progress_bar.value()
            if current < 100:
                self.progress_bar.setValue(current + 1)
            else:
                self.timer.stop()
    
    def process_recorded_audio(self, audio):
        """Process the recorded audio"""
        self.audio = audio
        self.is_recording = False
        self.record_button.setText("Record (5s)")
        self.classify_button.setEnabled(True)
        self.play_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Audio recorded. Click 'Classify Audio' to analyze.")
        
        # Visualize waveform
        self.waveform_canvas.plot_waveform(self.audio, SAMPLE_RATE)
        
        # Extract features
        self.extract_and_visualize_features()
    
    def load_audio(self):
        """Load audio from a file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav *.mp3 *.ogg);;All Files (*)"
        )
        
        if file_path:
            try:
                self.status_label.setText(f"Loading audio: {file_path}")
                
                # Load audio file
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Ensure max length
                if len(audio) > SAMPLE_RATE * MAX_AUDIO_LENGTH:
                    audio = audio[:SAMPLE_RATE * MAX_AUDIO_LENGTH]
                    
                self.audio = audio
                self.classify_button.setEnabled(True)
                self.play_button.setEnabled(True)
                self.status_label.setText(f"Loaded audio: {os.path.basename(file_path)}")
                
                # Visualize waveform
                self.waveform_canvas.plot_waveform(self.audio, SAMPLE_RATE)
                
                # Extract features
                self.extract_and_visualize_features()
                
            except Exception as e:
                self.status_label.setText(f"Error loading audio: {e}")
    
    def extract_and_visualize_features(self):
        """Extract and visualize features from the audio"""
        try:
            # Create a temporary file to use the feature extractor
            temp_file = 'temp_audio.wav'
            
            # Save the audio to a temporary file
            sf.write(temp_file, self.audio, SAMPLE_RATE)
            
            # Extract features
            self.features = extract_features(temp_file)
            
            # Clean up
            try:
                os.remove(temp_file)
            except:
                pass
            
            if self.features:
                # Visualize features
                self.spectrogram_canvas.plot_spectrogram(self.features['mel_spectrogram'])
                self.mfcc_canvas.plot_mfcc(self.features['mfcc'])
                self.pitch_canvas.plot_pitch(self.features['prosody'][0])  # F0
        except Exception as e:
            self.status_label.setText(f"Error extracting features: {e}")
    
    def play_audio(self):
        """Play the loaded audio"""
        if self.audio is not None:
            sd.play(self.audio, SAMPLE_RATE)
    
    def classify_audio(self):
        """Classify the audio as question or statement"""
        if self.model is None:
            self.status_label.setText("Error: Model not loaded")
            return
            
        if self.features is None:
            self.status_label.setText("Error: No features extracted")
            return
        
        try:
            # Ensure model is in eval mode
            self.model.eval()
            print(f"Model is on device: {next(self.model.parameters()).device}")
            
            # Convert features to tensors - use batch size of 2 for BatchNorm
            mel_spec = torch.tensor(self.features['mel_spectrogram'], dtype=torch.float)
            mel_spec = torch.stack([mel_spec, mel_spec])  # Create a batch of 2 identical examples
            
            mfcc = torch.tensor(self.features['mfcc'], dtype=torch.float)
            mfcc = torch.stack([mfcc, mfcc])  # Create a batch of 2
            
            prosody = torch.tensor(self.features['prosody'], dtype=torch.float)
            prosody = torch.stack([prosody, prosody])  # Create a batch of 2
            
            # Move tensors to the correct device
            mel_spec = mel_spec.to(self.device)
            mfcc = mfcc.to(self.device)
            prosody = prosody.to(self.device)
            
            print(f"Input tensors device: {mel_spec.device}")
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(mel_spec, mfcc, prosody)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                # Use only the first output (both outputs would be identical)
                probabilities = probabilities[0]
                _, predicted = torch.max(outputs[0:1], 1)
            
            # Get prediction and confidence
            pred_class = predicted.item()  # 0: Statement, 1: Question
            statement_prob = probabilities[0].item() * 100  # Convert to percentage
            question_prob = probabilities[1].item() * 100
            
            # Update UI
            if pred_class == 0:
                result_text = "Statement"
                self.result_label.setText(f"Result: {result_text}")
            else:
                result_text = "Question"
                self.result_label.setText(f"Result: {result_text}")
            
            # Update confidence bars
            self.statement_progress.setValue(int(statement_prob))
            self.question_progress.setValue(int(question_prob))
            
            self.status_label.setText(f"Classification complete: {result_text} (Statement: {statement_prob:.1f}%, Question: {question_prob:.1f}%)")
            
        except Exception as e:
            self.status_label.setText(f"Error during classification: {e}")
            import traceback
            traceback.print_exc()
    
    def load_sample_audio(self):
        """Load a sample audio file from the dataset"""
        try:
            dialog = SampleFilesDialog(self, sample_dir='swda_audio')
            if dialog.exec_() == QDialog.Accepted:
                file_path = dialog.get_selected_file()
                if file_path and os.path.exists(file_path):
                    self.status_label.setText(f"Loading sample audio: {os.path.basename(file_path)}")
                    
                    # Load audio file
                    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    
                    # Ensure max length
                    if len(audio) > SAMPLE_RATE * MAX_AUDIO_LENGTH:
                        audio = audio[:SAMPLE_RATE * MAX_AUDIO_LENGTH]
                        
                    self.audio = audio
                    self.classify_button.setEnabled(True)
                    self.play_button.setEnabled(True)
                    self.status_label.setText(f"Loaded sample: {os.path.basename(file_path)}")
                    
                    # Visualize waveform
                    self.waveform_canvas.plot_waveform(self.audio, SAMPLE_RATE)
                    
                    # Extract features
                    self.extract_and_visualize_features()
                else:
                    self.status_label.setText("No file selected or file not found")
        except Exception as e:
            self.status_label.setText(f"Error loading sample: {e}")
            QMessageBox.warning(self, "Error", f"Could not load sample: {e}")
    
    def handle_recording_error(self, error_msg):
        """Handle recording errors"""
        self.is_recording = False
        self.record_button.setText("Record (5s)")
        self.progress_bar.setVisible(False)
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        
        self.status_label.setText(f"Recording error: {error_msg}")
        QMessageBox.warning(self, "Recording Error", 
                          f"There was an error recording audio:\n{error_msg}\n\n" +
                          "Please try one of these alternatives:\n" +
                          "1. Use the 'Load Sample' button to select a pre-recorded audio file from the dataset\n" +
                          "2. Use the 'Load Audio File' button to select a .wav file from your computer\n" +
                          "3. Try a different audio device from the dropdown menu")
        
        # Focus on the device selection dropdown
        self.device_combo.setFocus()

if __name__ == "__main__":
    # Create application
    app = QApplication(sys.argv)
    
    # Apply stylesheet for better appearance
    app.setStyle("Fusion")
    
    # Dark mode palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    # Create and show window
    window = AudioClassifierApp()
    window.show()
    
    # Run application
    sys.exit(app.exec_()) 