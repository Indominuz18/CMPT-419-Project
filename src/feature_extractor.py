"""
Module for extracting features from audio for the SLUE dataset.
"""

import os
import numpy as np
import torch
import torchaudio
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class MelSpectrogramExtractor:
    """Class for extracting Mel Spectrogram features."""
    
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=400, hop_length=160):
        """
        Initialize Mel Spectrogram extractor.
        
        Args:
            sample_rate (int): Audio sample rate
            n_mels (int): Number of Mel bands
            n_fft (int): FFT window size
            hop_length (int): FFT hop length
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
    
    def __call__(self, waveform):
        """
        Extract Mel Spectrogram features.
        
        Args:
            waveform (torch.Tensor): Audio waveform
            
        Returns:
            torch.Tensor: Mel Spectrogram features
        """
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Extract features
        mel_spectrogram = self.transform(waveform)
        
        # Convert to log mel spectrogram
        log_mel_spectrogram = torch.log(mel_spectrogram + 1e-9)
        
        return log_mel_spectrogram

class Wav2Vec2FeatureExtractor:
    """Class for extracting features using pretrained Wav2Vec2 model."""
    
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device=None):
        """
        Initialize Wav2Vec2 feature extractor.
        
        Args:
            model_name (str): Pretrained model name
            device (str): Device to use ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def __call__(self, waveform, sample_rate=16000):
        """
        Extract features using Wav2Vec2.
        
        Args:
            waveform (torch.Tensor): Audio waveform
            sample_rate (int): Audio sample rate
            
        Returns:
            torch.Tensor: Extracted features
        """
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Convert to numpy and resample if needed
        waveform_np = waveform.squeeze().numpy()
        if sample_rate != 16000:
            waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=16000)
        
        # Process with Wav2Vec2
        inputs = self.processor(waveform_np, sampling_rate=16000, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get hidden states
        features = outputs.last_hidden_state
        
        return features.squeeze().cpu()

def process_dataset(data_loader, feature_extractor, output_dir, max_samples=None):
    """
    Process entire dataset and save features.
    
    Args:
        data_loader: DataLoader for the dataset
        feature_extractor: Feature extractor to use
        output_dir (str): Directory to save processed features
        max_samples (int, optional): Maximum number of samples to process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    for batch in data_loader:
        file_ids = batch['file_id']
        waveforms = batch['waveform']
        sample_rates = batch['sample_rate']
        labels = batch['label']
        
        for i, (file_id, waveform, sample_rate, label) in enumerate(zip(file_ids, waveforms, sample_rates, labels)):
            # Extract features
            features = feature_extractor(waveform, sample_rate)
            
            # Save features
            feature_path = os.path.join(output_dir, f"{file_id}.pt")
            torch.save({
                'features': features,
                'label': label
            }, feature_path)
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files")
                
            if max_samples is not None and processed_count >= max_samples:
                print(f"Reached maximum samples ({max_samples})")
                return
                
    print(f"Processed total of {processed_count} files")

if __name__ == '__main__':
    from data_loader import get_data_loader
    
    # Example usage
    data_dir = '../data/raw'
    output_dir = '../data/processed'
    
    # Initialize data loader
    train_loader = get_data_loader(data_dir, 'train', batch_size=1)
    
    # Initialize feature extractor
    feature_extractor = MelSpectrogramExtractor()
    
    # Process dataset
    process_dataset(train_loader, feature_extractor, output_dir, max_samples=10) 