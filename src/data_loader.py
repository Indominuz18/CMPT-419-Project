"""
Module for loading declarative question dataset from data_cleaning.
"""

import os
import json
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def pad_sequence(batch):
    """
    Pad a batch of tensors with different lengths.
    
    Args:
        batch: List of tensors
        
    Returns:
        torch.Tensor: Padded tensor
    """
    # Find maximum dimensions
    max_dims = []
    for dim in range(batch[0].dim()):
        max_dim = max([x.size(dim) for x in batch])
        max_dims.append(max_dim)
    
    # Pad each tensor to the maximum dimensions
    padded_batch = []
    for tensor in batch:
        pad_sizes = []
        for dim in range(tensor.dim()):
            pad_size = max_dims[dim] - tensor.size(dim)
            # Add padding at the end of each dimension
            pad_sizes.extend([0, pad_size])
        
        # Reverse pad_sizes because F.pad expects dimensions in reverse order
        padded_tensor = F.pad(tensor, pad_sizes[::-1])
        padded_batch.append(padded_tensor)
    
    return torch.stack(padded_batch)

def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable-length tensors.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        dict: Collated batch
    """
    # Extract keys from the first sample
    keys = batch[0].keys()
    result = {key: [] for key in keys}
    
    # Group items by key
    for sample in batch:
        for key in keys:
            result[key].append(sample[key])
    
    # Process each key
    for key in keys:
        if key == 'waveform' or key == 'features':
            # Handle features dictionary
            if isinstance(result[key][0], dict):
                feature_types = result[key][0].keys()
                padded_features = {}
                
                for feature_type in feature_types:
                    # Extract features of this type from all samples
                    features = [sample[feature_type] for sample in result[key]]
                    
                    # Check if all tensors have the same shape
                    shapes = [f.shape for f in features]
                    if len(set(str(s) for s in shapes)) == 1:
                        # All same shape, use regular stack
                        padded_features[feature_type] = torch.stack(features)
                    else:
                        # Different shapes, use padding
                        padded_features[feature_type] = pad_sequence(features)
                
                result[key] = padded_features
            # Handle tensor features
            elif torch.is_tensor(result[key][0]):
                shapes = [t.shape for t in result[key]]
                if len(set(str(s) for s in shapes)) == 1:
                    # All same shape, use regular stack
                    result[key] = torch.stack(result[key])
                else:
                    # Different shapes, use padding
                    result[key] = pad_sequence(result[key])
        elif key == 'label':
            # Convert labels to tensor
            result[key] = torch.tensor(result[key])
        elif key == 'sample_rate':
            # Convert to tensor if all values are the same
            if len(set(result[key])) == 1:
                result[key] = torch.tensor([result[key][0]] * len(result[key]))
        # Keep other keys as lists
    
    return result

class DeclarativeQuestionDataset(Dataset):
    """Dataset class for declarative question dataset."""
    
    def __init__(self, data_dir, csv_path, split='train', transform=None, label_map=None):
        """
        Initialize dataset.
        
        Args:
            data_dir (str): Directory containing audio files (split_audio or split_audiov2)
            csv_path (str): Path to the swda_declarative.csv file
            split (str): Data split ('train', 'dev', or 'test')
            transform (callable, optional): Optional transform to be applied on a sample
            label_map (dict, optional): Mapping from string labels to numeric indices
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load CSV data
        self.df = pd.read_csv(csv_path)
        
        # Filter out rows without audio files
        self.df = self.df[self.df['found'].notna() & (self.df['found'] != '')]
        
        # Create label map if not provided
        if label_map is None:
            # Default for binary classification (declarative question or not)
            self.label_map = {'qy^d': 1, 'non-qy^d': 0}
        else:
            self.label_map = label_map
            
        # Process file paths based on 'found' column
        self.audio_files = []
        self.labels = []
        
        for _, row in self.df.iterrows():
            # Handle multiple audio files separated by commas
            if isinstance(row['found'], str) and row['found'].strip():
                file_paths = [path.strip() for path in row['found'].split(',') if path.strip()]
                for path in file_paths:
                    if os.path.exists(os.path.join(self.data_dir, path)):
                        self.audio_files.append(path)
                        
                        # Extract label - assuming meta.tag contains the label info
                        if 'qy^d' in str(row['meta.tag']):
                            self.labels.append(self.label_map['qy^d'])
                        else:
                            self.labels.append(self.label_map['non-qy^d'])
        
        # As a fallback, also scan the folders for audio files
        self._scan_folders_for_additional_files()
    
    def _scan_folders_for_additional_files(self):
        """Scan folders for additional audio files from MRDA corpus."""
        # This handles files produced by mrda.py
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('_audio.wav') and os.path.join(root, file) not in self.audio_files:
                    rel_path = os.path.relpath(os.path.join(root, file), self.data_dir)
                    self.audio_files.append(rel_path)
                    # For MRDA corpus, all extracted files are declarative questions (qy^d)
                    self.labels.append(self.label_map['qy^d'])
    
    def __len__(self):
        """Return the number of samples in dataset."""
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: A dictionary containing the audio and labels
        """
        audio_path = os.path.join(self.data_dir, self.audio_files[idx])
        label = self.labels[idx]
        
        # Load audio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return a dummy tensor on error
            waveform = torch.zeros(1, 16000)
            sample_rate = 16000
        
        sample = {
            'file_id': self.audio_files[idx],
            'waveform': waveform,
            'sample_rate': sample_rate,
            'label': label
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def get_data_loader(data_dir, csv_path, split='train', batch_size=32, num_workers=4, transform=None, label_map=None):
    """
    Create a data loader for the specified split.
    
    Args:
        data_dir (str): Directory containing audio files
        csv_path (str): Path to the swda_declarative.csv file
        split (str): Data split ('train', 'dev', or 'test')
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        transform (callable, optional): Optional transform to be applied on a sample
        label_map (dict, optional): Mapping from string labels to numeric indices
        
    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    dataset = DeclarativeQuestionDataset(data_dir, csv_path, split, transform, label_map)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    return loader

class SLUEDataset(Dataset):
    """Original SLUE dataset class (kept for backward compatibility)."""
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Initialize dataset.
        
        Args:
            data_dir (str): Directory containing data files
            split (str): Data split ('train', 'dev', or 'test')
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.metadata_path = os.path.join(data_dir, f"{split}_metadata.json")
        
        # Load metadata if file exists
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.file_ids = list(self.metadata.keys())
        else:
            # Fallback to DeclarativeQuestionDataset if metadata file doesn't exist
            print(f"Warning: {self.metadata_path} not found. Using DeclarativeQuestionDataset instead.")
            csv_path = os.path.join(os.path.dirname(data_dir), "data_cleaning/swda_declarative.csv")
            self.dq_dataset = DeclarativeQuestionDataset(
                os.path.join(os.path.dirname(data_dir), "data_cleaning/split_audiov2"), 
                csv_path, 
                split, 
                transform
            )
    
    def __len__(self):
        """Return the number of samples in dataset."""
        if hasattr(self, 'file_ids'):
            return len(self.file_ids)
        else:
            return len(self.dq_dataset)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: A dictionary containing the audio and labels
        """
        if hasattr(self, 'file_ids'):
            file_id = self.file_ids[idx]
            metadata = self.metadata[file_id]
            
            # Load audio
            audio_path = os.path.join(self.data_dir, 'audio', f"{file_id}.wav")
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Get label
            label = metadata['label']
            
            sample = {
                'file_id': file_id,
                'waveform': waveform,
                'sample_rate': sample_rate,
                'label': label,
                'metadata': metadata
            }
            
            if self.transform:
                sample = self.transform(sample)
                
            return sample
        else:
            return self.dq_dataset[idx]

if __name__ == '__main__':
    # Example usage
    data_dir = '../data_cleaning/split_audiov2'
    csv_path = '../data_cleaning/swda_declarative.csv'
    
    # Get data loader with new Dataset class
    train_loader = get_data_loader(data_dir, csv_path, 'train')
    
    # Print sample data
    for batch in train_loader:
        print(f"Batch size: {len(batch['file_id'])}")
        print(f"Waveform shape: {batch['waveform'].shape}")
        print(f"Sample rate: {batch['sample_rate'][0]}")
        print(f"Labels: {batch['label']}")
        break 