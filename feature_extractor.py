import os
import numpy as np
import librosa
import librosa.display
import pickle
import re
from tqdm import tqdm
import random
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
import glob
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Audio settings
SAMPLE_RATE = 16000  # Common sample rate for speech
MAX_AUDIO_LENGTH = 8  # Maximum audio length in seconds
NUM_MELS = 128
NUM_MFCC = 40

# Extract conversation ID and speaker from filename
def extract_metadata(filename):
    # Pattern: sw[conversation_id][speaker]_[start_time]_[end_time].wav
    pattern = r'sw(\d+)([AB])_(\d+)_(\d+)'
    match = re.match(pattern, filename)
    if match:
        conversation_id, speaker, start_time, end_time = match.groups()
        return conversation_id, speaker
    return None, None

# Enhanced question detection with better heuristics
def determine_label(audio_path):
    """Determine if audio is a question or statement using more sophisticated acoustic cues."""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Apply pre-emphasis filter to enhance high frequencies
        y = librosa.effects.preemphasis(y)
        
        # Ensure audio is at least 1 second long
        if len(y) < sr:
            return 0  # Default to statement if too short
        
        # Extract pitch contour with higher accuracy
        hop_length = 256
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=hop_length,
            fill_na=None
        )
        
        # Replace NaN values with 0
        f0 = np.nan_to_num(f0)
        
        # Apply median filtering to smooth the contour
        f0_smooth = signal.medfilt(f0, kernel_size=5)
        
        # Calculate pitch rise (key indicator of questions)
        if len(f0_smooth) > sr // hop_length:
            # Analyze pitch contour in segments
            segments = 5  # Divide into 5 segments
            segment_size = len(f0_smooth) // segments
            
            # Get average pitch for each segment (ignoring zeros)
            segment_pitches = []
            for i in range(segments):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < segments - 1 else len(f0_smooth)
                segment = f0_smooth[start_idx:end_idx]
                valid_pitches = segment[segment > 0]
                if len(valid_pitches) > 0:
                    segment_pitches.append(np.mean(valid_pitches))
                else:
                    segment_pitches.append(0)
            
            # Calculate pitch trend features
            if all(p > 0 for p in segment_pitches):
                # Pitch rise at the end (last segment vs. first segment)
                final_rise = segment_pitches[-1] / segment_pitches[0] if segment_pitches[0] > 0 else 1
                
                # Check if pitch rises in the last two segments
                late_rise = segment_pitches[-1] / segment_pitches[-2] if segment_pitches[-2] > 0 else 1
                
                # Pitch variability
                pitch_range = np.max(segment_pitches) / np.min(segment_pitches) if np.min(segment_pitches) > 0 else 1
                pitch_std = np.std(segment_pitches) / np.mean(segment_pitches) if np.mean(segment_pitches) > 0 else 0
            else:
                final_rise = 1
                late_rise = 1
                pitch_range = 1
                pitch_std = 0
        else:
            final_rise = 1
            late_rise = 1
            pitch_range = 1
            pitch_std = 0
        
        # Extract energy contour
        energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        energy_smooth = signal.medfilt(energy, kernel_size=5)
        
        # Calculate energy patterns
        if len(energy_smooth) > 3:
            # Divide energy into segments
            segments = 5
            segment_size = len(energy_smooth) // segments
            
            # Get average energy for each segment
            segment_energies = []
            for i in range(segments):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < segments - 1 else len(energy_smooth)
                segment_energies.append(np.mean(energy_smooth[start_idx:end_idx]))
            
            # Energy patterns
            final_energy_ratio = segment_energies[-1] / segment_energies[0] if segment_energies[0] > 0 else 1
            energy_std = np.std(segment_energies) / np.mean(segment_energies) if np.mean(segment_energies) > 0 else 0
        else:
            final_energy_ratio = 1
            energy_std = 0
        
        # Timing/rhythm features
        duration = len(y) / sr
        short_utterance = duration < 2.0  # Questions often shorter
        
        # Spectral features for voice quality
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Calculate spectral statistics
        high_spectral_centroid = np.mean(spectral_centroid) > 2000  # Higher spectral centroid often in questions
        
        # Formant analysis approximation (using MFCC variance in regions)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        formant_variability = np.mean(np.var(mfccs[1:5], axis=1))  # Variability in first few formants
        high_formant_var = formant_variability > 15
        
        # Voice quality - zero crossing rate for breathiness
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        high_zcr = np.mean(zcr) > 0.1  # Higher ZCR can indicate question intonation
        
        # Combined decision logic - weighted scoring
        # Base score starts at 0
        score = 0
        
        # Strong indicators (pitch patterns)
        if final_rise > 1.25:  # Final pitch rise of 25% or more
            score += 3
        if late_rise > 1.15:  # Rise in the last segment
            score += 2
        
        # Moderate indicators
        if pitch_range > 1.5:  # Wide pitch range
            score += 1
        if pitch_std > 0.2:  # High pitch variability
            score += 1
        if final_energy_ratio < 0.8:  # Energy drop at the end
            score += 1
        if high_spectral_centroid:  # Higher frequencies
            score += 1
        if high_formant_var:  # Formant variability
            score += 1
        
        # Weak indicators
        if short_utterance:  # Short duration
            score += 0.5
        if energy_std > 0.3:  # Energy variability
            score += 0.5
        if high_zcr:  # Breathiness
            score += 0.5
        
        # Final decision threshold
        return 1 if score >= 3.5 else 0
        
    except Exception as e:
        print(f"Error in determine_label for {audio_path}: {e}")
        return 0  # Default to statement if processing fails

# Advanced feature extraction
def extract_melspectrogram(audio, sr):
    """Extract enhanced mel spectrogram with better parameters."""
    # Apply pre-emphasis to highlight higher frequencies
    audio_preemph = librosa.effects.preemphasis(audio)
    
    # Extract mel spectrogram with improved parameters
    mel_spec = librosa.feature.melspectrogram(
        y=audio_preemph, 
        sr=sr, 
        n_mels=NUM_MELS, 
        n_fft=2048,
        hop_length=512,
        fmin=50,  # Focus more on speech range
        fmax=8000,
        power=2.0  # Square the magnitude
    )
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Apply per-channel energy normalization (PCEN)
    pcen = librosa.pcen(mel_spec, sr=sr)
    
    # Combine log-mel and PCEN representations
    combined_mel = np.stack([log_mel_spec, pcen])
    
    # Normalize to [-1, 1] range
    combined_mel = (combined_mel - np.mean(combined_mel)) / (np.std(combined_mel) + 1e-9)
    
    return combined_mel

def extract_mfcc(audio, sr):
    """Extract MFCC features with enhanced parameters."""
    # Apply pre-emphasis
    audio_preemph = librosa.effects.preemphasis(audio)
    
    # Extract MFCCs with improved parameters
    mfcc = librosa.feature.mfcc(
        y=audio_preemph, 
        sr=sr, 
        n_mfcc=NUM_MFCC,
        n_fft=2048,
        hop_length=512,
        lifter=22,
        htk=True
    )
    
    # Add velocity and acceleration
    delta_mfcc = librosa.feature.delta(mfcc, width=9)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=9)
    
    # Calculate statistics over time
    mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
    mfcc_std = np.std(mfcc, axis=1, keepdims=True)
    mfcc_skew = np.array([skew(row) for row in mfcc]).reshape(-1, 1)
    
    # Stack features
    mfcc_features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)
    
    # Normalize each feature set
    for i in range(mfcc_features.shape[0]):
        mfcc_features[i] = (mfcc_features[i] - np.mean(mfcc_features[i])) / (np.std(mfcc_features[i]) + 1e-9)
    
    return mfcc_features

def extract_prosody(audio, sr):
    """Extract enhanced prosodic features."""
    hop_length = 512
    
    # F0 (pitch) using PYIN algorithm with better parameters
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=hop_length,
        fill_na=None
    )
    
    # Replace NaN values with zeros
    f0 = np.nan_to_num(f0)
    
    # Smooth pitch contour
    f0_smooth = signal.medfilt(f0, kernel_size=5)
    
    # Energy (RMS)
    rmse = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    
    # Zero crossing rate (for voice/unvoiced detection)
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
    
    # Spectral centroid (brightness of sound)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    
    # Spectral flux (rate of change of spectrum)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    
    # Spectral contrast (formant-like representation)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=hop_length)
    contrast_mean = np.mean(contrast, axis=1)
    
    # Harmonic-percussive source separation for cleaner voice analysis
    y_harmonic, y_percussive = librosa.effects.hpss(audio)
    
    # Harmonics-to-noise ratio approximation
    hnr = np.mean(y_harmonic) / (np.mean(y_percussive) + 1e-9)
    hnr_array = np.ones_like(f0_smooth) * hnr
    
    # Speech rate approximation using onset strength
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=hop_length)
    speech_rate = len(onsets) / (len(audio) / sr)
    speech_rate_array = np.ones_like(f0_smooth) * speech_rate
    
    # Calculate pitch contour features
    if len(f0_smooth) > 0:
        # Pitch statistics
        voiced_f0 = f0_smooth[f0_smooth > 0]
        if len(voiced_f0) > 0:
            f0_range = np.max(voiced_f0) - np.min(voiced_f0)
            f0_mean = np.mean(voiced_f0)
            f0_std = np.std(voiced_f0)
        else:
            f0_range = 0
            f0_mean = 0
            f0_std = 0
            
        # Calculate pitch slope features
        if len(voiced_f0) > 10:
            # Divide into segments
            segments = min(5, len(voiced_f0) // 5)
            segment_size = len(voiced_f0) // segments
            
            pitch_slopes = []
            for i in range(segments):
                if i < segments - 1:
                    segment = voiced_f0[i*segment_size:(i+1)*segment_size]
                else:
                    segment = voiced_f0[i*segment_size:]
                
                if len(segment) > 1:
                    # Calculate slope
                    x = np.arange(len(segment))
                    slope = np.polyfit(x, segment, 1)[0]
                    pitch_slopes.append(slope)
                else:
                    pitch_slopes.append(0)
            
            # Create arrays of pitch slopes
            pitch_slope_array = np.zeros_like(f0_smooth)
            for i, slope in enumerate(pitch_slopes):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, len(pitch_slope_array))
                pitch_slope_array[start_idx:end_idx] = slope
        else:
            pitch_slope_array = np.zeros_like(f0_smooth)
    else:
        pitch_slope_array = np.zeros(1)
    
    # Feature arrays to stack
    feature_arrays = [
        f0_smooth,                # Pitch contour
        rmse,                     # Energy contour
        zcr,                      # Voice/unvoiced detection
        spectral_centroid,        # Spectral brightness
        onset_env,                # Rhythm/changes
        pitch_slope_array,        # Pitch trends
        hnr_array,                # Voice quality
        speech_rate_array         # Speech rate
    ]
    
    # Reshape to consistent dimensions
    max_len = max(len(arr) for arr in feature_arrays)
    padded_features = []
    
    for feat in feature_arrays:
        pad_width = max_len - len(feat)
        if pad_width > 0:
            padded = np.pad(feat, (0, pad_width), mode='constant')
        else:
            padded = feat[:max_len]
        padded_features.append(padded)
    
    # Stack and normalize features
    prosody_features = np.vstack(padded_features)
    
    # Normalize each feature
    for i in range(prosody_features.shape[0]):
        if np.std(prosody_features[i]) > 0:
            prosody_features[i] = (prosody_features[i] - np.mean(prosody_features[i])) / (np.std(prosody_features[i]) + 1e-9)
    
    # Keep top features (most relevant for question detection)
    return prosody_features

# Data augmentation functions
def time_stretch(audio, rate_range=(0.9, 1.1)):
    """Apply time stretching augmentation."""
    rate = random.uniform(rate_range[0], rate_range[1])
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr, semitones_range=(-2, 2)):
    """Apply pitch shifting augmentation."""
    semitones = random.uniform(semitones_range[0], semitones_range[1])
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)

def add_noise(audio, noise_level_range=(0.001, 0.005)):
    """Add random noise to the audio."""
    noise_level = random.uniform(noise_level_range[0], noise_level_range[1])
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise

def apply_augmentation(audio, sr, augment_type=None):
    """Apply random augmentation to the audio."""
    if augment_type is None:
        # Choose random augmentation
        augment_type = random.choice(['time', 'pitch', 'noise', 'none'])
    
    if augment_type == 'time':
        return time_stretch(audio)
    elif augment_type == 'pitch':
        return pitch_shift(audio, sr)
    elif augment_type == 'noise':
        return add_noise(audio)
    else:  # 'none' or unknown type
        return audio

# Function to extract all features from an audio file
def extract_features(audio_path, augment=False):
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Apply augmentation if requested
        if augment:
            y = apply_augmentation(y, sr)
        
        # Ensure consistent length
        if len(y) > SAMPLE_RATE * MAX_AUDIO_LENGTH:
            y = y[:SAMPLE_RATE * MAX_AUDIO_LENGTH]
        else:
            # Use reflection padding instead of zeros
            y = np.pad(y, (0, max(0, SAMPLE_RATE * MAX_AUDIO_LENGTH - len(y))), mode='reflect')
        
        # Apply pre-emphasis
        y = librosa.effects.preemphasis(y)
        
        # Extract features
        mel_spec = extract_melspectrogram(y, sr)
        mfcc = extract_mfcc(y, sr)
        prosody = extract_prosody(y, sr)
        
        return {
            'mel_spectrogram': mel_spec,
            'mfcc': mfcc,
            'prosody': prosody
        }
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def process_dataset(audio_dir='swda_audio', output_dir='processed_data', augment=True):
    """Process all audio files and save features and labels to disk."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all audio file paths
    audio_paths = []
    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            audio_paths.append(os.path.join(audio_dir, filename))
    
    print(f"Found {len(audio_paths)} audio files")
    
    # Create labels
    labels = []
    features_list = []
    filenames = []
    augmented_count = 0
    
    # Process all files with progress bar
    for path in tqdm(audio_paths, desc="Extracting features"):
        # Determine label
        label = determine_label(path)
        
        # Extract features for original audio
        features = extract_features(path, augment=False)
        
        if features is not None:
            labels.append(label)
            features_list.append(features)
            filenames.append(os.path.basename(path))
            
            # Apply data augmentation for minority class
            if augment and label == 1:  # If it's a question
                # Add multiple augmented versions for better balance
                for aug_type in ['time', 'pitch', 'noise']:
                    aug_features = extract_features(path, augment=True)
                    if aug_features is not None:
                        labels.append(label)
                        features_list.append(aug_features)
                        filenames.append(f"aug_{aug_type}_{os.path.basename(path)}")
                        augmented_count += 1
    
    # Save processed data
    data = {
        'features': features_list,
        'labels': labels,
        'filenames': filenames
    }
    
    with open(os.path.join(output_dir, 'processed_features.pkl'), 'wb') as f:
        pickle.dump(data, f)
    
    # Count classes
    question_count = sum(labels)
    statement_count = len(labels) - question_count
    
    print(f"Processed {len(audio_paths)} original files successfully")
    print(f"Added {augmented_count} augmented samples")
    print(f"Total dataset size: {len(features_list)} samples")
    print(f"Data saved to {os.path.join(output_dir, 'processed_features.pkl')}")
    
    # Return basic statistics
    return {
        'total': len(labels),
        'questions': question_count,
        'statements': statement_count,
        'question_ratio': question_count / len(labels) if len(labels) > 0 else 0,
        'augmented_count': augmented_count
    }

if __name__ == "__main__":
    stats = process_dataset()
    print("\nDataset Statistics:")
    print(f"Total samples: {stats['total']}")
    print(f"Questions: {stats['questions']} ({stats['question_ratio']:.2%})")
    print(f"Statements: {stats['statements']} ({1-stats['question_ratio']:.2%})")
    print(f"Augmented samples: {stats['augmented_count']}")

    # Generate and save sample visualizations
    num_samples = min(5, len(stats['features']))
    sample_indices = np.random.choice(len(stats['features']), num_samples, replace=False)
    
    for idx in sample_indices:
        label = "Question" if stats['labels'][idx] == 1 else "Statement"
        filename = stats['filenames'][idx]
        feature = stats['features'][idx]
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot Mel Spectrogram
        librosa.display.specshow(
            feature['mel_spectrogram'][0], x_axis='time', y_axis='mel', sr=SAMPLE_RATE,
            fmax=8000, ax=axes[0], cmap='viridis'
        )
        axes[0].set_title(f'{label} - {filename} - Mel Spectrogram')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Mel Frequency')
        
        # Plot MFCC
        librosa.display.specshow(
            feature['mfcc'][:NUM_MFCC], x_axis='time', ax=axes[1], cmap='viridis'
        )
        axes[1].set_title(f'{label} - {filename} - MFCC')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('MFCC Coefficients')
        
        # Plot Prosodic Features
        time = np.linspace(0, MAX_AUDIO_LENGTH, feature['prosody'].shape[1])
        axes[2].plot(time, feature['prosody'][0], label='Pitch', alpha=0.8)
        axes[2].plot(time, feature['prosody'][3], label='Energy', alpha=0.8)
        axes[2].plot(time, feature['prosody'][7], label='Terminal Rise', alpha=0.8)
        axes[2].set_title(f'{label} - {filename} - Prosodic Features')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Normalized Amplitude')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{idx}_{label}.png'))
        plt.close(fig)
    
    print(f"Generated {num_samples} sample visualizations in {output_dir}") 