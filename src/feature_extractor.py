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
    
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=1024, hop_length=160):
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

class ProsodyFeatureExtractor:
    """Class for extracting prosodic features from audio."""
    
    def __init__(self, sample_rate=16000, hop_length=160, f0_min=60, f0_max=500):
        """
        Initialize prosody feature extractor.
        
        Args:
            sample_rate (int): Audio sample rate
            hop_length (int): Hop length for feature extraction
            f0_min (int): Minimum F0 frequency to detect
            f0_max (int): Maximum F0 frequency to detect
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
    
    def __call__(self, waveform, sample_rate=16000):
        """
        Extract prosodic features.
        
        Args:
            waveform (torch.Tensor): Audio waveform
            sample_rate (int): Audio sample rate
            
        Returns:
            dict: Dictionary containing prosodic features
        """
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Convert to numpy and resample if needed
        waveform_np = waveform.squeeze().numpy()
        if sample_rate != self.sample_rate:
            waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=self.sample_rate)
        
        # Extract pitch (fundamental frequency)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            waveform_np, 
            fmin=self.f0_min,
            fmax=self.f0_max,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Calculate pitch statistics
        f0_nonzero = f0[f0 > 0]
        pitch_stats = {
            'mean': np.mean(f0_nonzero) if len(f0_nonzero) > 0 else 0,
            'std': np.std(f0_nonzero) if len(f0_nonzero) > 0 else 0,
            'min': np.min(f0_nonzero) if len(f0_nonzero) > 0 else 0,
            'max': np.max(f0_nonzero) if len(f0_nonzero) > 0 else 0,
            'range': np.ptp(f0_nonzero) if len(f0_nonzero) > 0 else 0,
        }
        
        # Extract energy/intensity
        rms = librosa.feature.rms(y=waveform_np, hop_length=self.hop_length)[0]
        energy_stats = {
            'mean': np.mean(rms),
            'std': np.std(rms),
            'min': np.min(rms),
            'max': np.max(rms),
            'range': np.ptp(rms),
        }
        
        # Extract speaking rate (using zero-crossing rate as proxy)
        zcr = librosa.feature.zero_crossing_rate(waveform_np, hop_length=self.hop_length)[0]
        speaking_rate_stats = {
            'mean': np.mean(zcr),
            'std': np.std(zcr),
        }
        
        # Extract voice quality features (spectral centroid and spectral contrast)
        centroid = librosa.feature.spectral_centroid(y=waveform_np, sr=self.sample_rate, hop_length=self.hop_length)[0]
        contrast = librosa.feature.spectral_contrast(y=waveform_np, sr=self.sample_rate, hop_length=self.hop_length)
        
        voice_quality_stats = {
            'centroid_mean': np.mean(centroid),
            'centroid_std': np.std(centroid),
            'contrast_mean': np.mean(contrast),
            'contrast_std': np.std(contrast),
        }
        
        # Combine all features
        features = {
            'pitch': pitch_stats,
            'energy': energy_stats,
            'speaking_rate': speaking_rate_stats,
            'voice_quality': voice_quality_stats,
            'raw_f0': f0,
            'raw_energy': rms,
        }
        
        # Convert to tensor
        features_flat = torch.tensor([
            pitch_stats['mean'], pitch_stats['std'], pitch_stats['range'],
            energy_stats['mean'], energy_stats['std'], energy_stats['range'],
            speaking_rate_stats['mean'], speaking_rate_stats['std'],
            voice_quality_stats['centroid_mean'], voice_quality_stats['centroid_std'],
            voice_quality_stats['contrast_mean'], voice_quality_stats['contrast_std']
        ], dtype=torch.float32)
        
        return features_flat

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

class CombinedFeatureExtractor:
    """Class for combining multiple feature extractors."""
    
    def __init__(self, feature_extractors):
        """
        Initialize combined feature extractor.
        
        Args:
            feature_extractors (list): List of feature extractors
        """
        self.feature_extractors = feature_extractors
    
    def __call__(self, waveform, sample_rate=16000):
        """
        Extract features using multiple extractors.
        
        Args:
            waveform (torch.Tensor): Audio waveform
            sample_rate (int): Audio sample rate
            
        Returns:
            list: List of extracted features
        """
        features = []
        for extractor in self.feature_extractors:
            features.append(extractor(waveform, sample_rate))
        
        return features

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
    # feature_extractor = MelSpectrogramExtractor()
    feature_extractor = ProsodyFeatureExtractor()
    
    # Process dataset
    process_dataset(train_loader, feature_extractor, output_dir, max_samples=10) 