import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import librosa.display
import warnings
from classify_audio_cnn_lstm import classify_audio_file

# Suppress warnings
warnings.filterwarnings("ignore")

class AudioClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Statement vs. Question Audio Classifier")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Model paths
        self.model_path = "./output/cnn_lstm_model.pth"
        self.scaler_path = "./output/cnn_lstm_scaler.pkl"
        
        # Check GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create UI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#f0f0f0')
        header_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(
            header_frame, 
            text="Audio Statement vs. Question Classifier",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0'
        ).pack(side=tk.LEFT)
        
        device_text = f"Using: {'GPU' if torch.cuda.is_available() else 'CPU'}"
        tk.Label(
            header_frame, 
            text=device_text,
            font=("Arial", 10),
            bg='#f0f0f0'
        ).pack(side=tk.RIGHT)
        
        # Main content frame
        content_frame = tk.Frame(self.root, bg='#f0f0f0')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel for controls
        control_frame = tk.Frame(content_frame, bg='#f0f0f0', width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Select audio file
        tk.Label(
            control_frame, 
            text="Step 1: Select Audio File",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0'
        ).pack(anchor=tk.W, pady=(0, 5))
        
        self.select_btn = tk.Button(
            control_frame,
            text="Browse...",
            command=self.select_audio_file,
            bg='#4CAF50',
            fg='white',
            font=("Arial", 10),
            padx=10
        )
        self.select_btn.pack(anchor=tk.W, pady=5)
        
        self.file_label = tk.Label(
            control_frame,
            text="No file selected",
            font=("Arial", 10),
            bg='#f0f0f0',
            wraplength=180
        )
        self.file_label.pack(anchor=tk.W, pady=5)
        
        # Classify button
        tk.Label(
            control_frame, 
            text="Step 2: Classify Audio",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0'
        ).pack(anchor=tk.W, pady=(20, 5))
        
        self.classify_btn = tk.Button(
            control_frame,
            text="Classify Audio",
            command=self.classify_audio,
            bg='#2196F3',
            fg='white',
            font=("Arial", 10),
            padx=10,
            state=tk.DISABLED
        )
        self.classify_btn.pack(anchor=tk.W, pady=5)
        
        # Results section
        tk.Label(
            control_frame, 
            text="Results:",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0'
        ).pack(anchor=tk.W, pady=(20, 5))
        
        self.result_label = tk.Label(
            control_frame,
            text="",
            font=("Arial", 12),
            bg='#f0f0f0'
        )
        self.result_label.pack(anchor=tk.W, pady=5)
        
        self.confidence_label = tk.Label(
            control_frame,
            text="",
            font=("Arial", 10),
            bg='#f0f0f0'
        )
        self.confidence_label.pack(anchor=tk.W, pady=5)
        
        self.explanation_text = tk.Text(
            control_frame,
            height=6,
            width=25,
            wrap=tk.WORD,
            font=("Arial", 10),
            bg='#f5f5f5',
            relief=tk.FLAT
        )
        self.explanation_text.pack(anchor=tk.W, pady=5, fill=tk.X)
        self.explanation_text.config(state=tk.DISABLED)
        
        # Right panel for visualization
        viz_frame = tk.Frame(content_frame, bg='#ffffff', padx=10, pady=10)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create figure for plots
        self.fig = plt.Figure(figsize=(6, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize subplots
        self.waveform_ax = self.fig.add_subplot(3, 1, 1)
        self.mel_ax = self.fig.add_subplot(3, 1, 2)
        self.pitch_ax = self.fig.add_subplot(3, 1, 3)
        
        self.fig.tight_layout(pad=3.0)
        
        # Footer
        footer_frame = tk.Frame(self.root, bg='#f0f0f0')
        footer_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(
            footer_frame, 
            text="CNN+LSTM Model with MEL Spectrograms & Prosodic Features",
            font=("Arial", 8),
            bg='#f0f0f0'
        ).pack(side=tk.LEFT)
    
    def select_audio_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            self.audio_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
            self.classify_btn.config(state=tk.NORMAL)
            self.display_audio(file_path)
        
    def display_audio(self, audio_path):
        # Load audio
        try:
            y, sr = librosa.load(audio_path, sr=None)
            
            # Clear previous plots
            self.waveform_ax.clear()
            self.mel_ax.clear()
            self.pitch_ax.clear()
            
            # Plot waveform
            self.waveform_ax.set_title("Waveform")
            librosa.display.waveshow(y, sr=sr, ax=self.waveform_ax)
            
            # Plot mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            img = librosa.display.specshow(
                mel_db, 
                x_axis='time', 
                y_axis='mel', 
                sr=sr, 
                ax=self.mel_ax
            )
            self.mel_ax.set_title("MEL Spectrogram")
            
            # Extract pitch and plot
            f0, voiced_flag, _ = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            
            times = librosa.times_like(f0, sr=sr)
            self.pitch_ax.plot(times, f0, label='F0', linewidth=2)
            self.pitch_ax.set_title("Pitch Contour (F0)")
            self.pitch_ax.set_xlabel("Time (s)")
            self.pitch_ax.set_ylabel("Frequency (Hz)")
            
            # Update canvas
            self.fig.tight_layout(pad=3.0)
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not display audio: {str(e)}")
    
    def classify_audio(self):
        try:
            # Reset result display
            self.result_label.config(text="Classifying...", fg="black")
            self.confidence_label.config(text="")
            self.explanation_text.config(state=tk.NORMAL)
            self.explanation_text.delete(1.0, tk.END)
            self.explanation_text.config(state=tk.DISABLED)
            self.root.update()
            
            # Classify audio
            result = classify_audio_file(
                self.audio_path,
                self.model_path,
                self.scaler_path
            )
            
            if result:
                prediction, confidence = result
                
                # Update results
                self.result_label.config(
                    text=f"Prediction: {prediction}",
                    fg="#2196F3" if prediction == "Question" else "#4CAF50"
                )
                self.confidence_label.config(
                    text=f"Confidence: {confidence:.2f}"
                )
                
                # Add explanation
                self.explanation_text.config(state=tk.NORMAL)
                self.explanation_text.delete(1.0, tk.END)
                
                if prediction == "Question":
                    explanation = ("This utterance likely has rising intonation " +
                                  "at the end, which is characteristic of " +
                                  "questions in English and many other languages.")
                else:
                    explanation = ("This utterance likely has falling or flat " +
                                  "intonation, which is characteristic of " +
                                  "statements or declarations.")
                
                self.explanation_text.insert(tk.END, explanation)
                self.explanation_text.config(state=tk.DISABLED)
            else:
                self.result_label.config(
                    text="Classification failed",
                    fg="red"
                )
        
        except Exception as e:
            messagebox.showerror("Error", f"Classification error: {str(e)}")
            self.result_label.config(
                text="Classification failed",
                fg="red"
            )

def main():
    # Create root window and app
    root = tk.Tk()
    app = AudioClassifierApp(root)
    
    # Check if model exists
    if not os.path.exists("./output/cnn_lstm_model.pth"):
        messagebox.showwarning(
            "Model Not Found", 
            "The model file was not found. Please train the model first by running:\n\npython cnn_lstm_classifier.py"
        )
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main() 