"""
Module for loading SLUE dataset.
"""

import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class SLUEDataset(Dataset):
    """Dataset class for SLUE dataset."""
    
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
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.file_ids = list(self.metadata.keys())
    
    def __len__(self):
        """Return the number of samples in dataset."""
        return len(self.file_ids)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: A dictionary containing the audio and labels
        """
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

def get_data_loader(data_dir, split='train', batch_size=32, num_workers=4, transform=None):
    """
    Create a data loader for the specified split.
    
    Args:
        data_dir (str): Directory containing data files
        split (str): Data split ('train', 'dev', or 'test')
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        transform (callable, optional): Optional transform to be applied on a sample
        
    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    dataset = SLUEDataset(data_dir, split, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    return loader

if __name__ == '__main__':
    # Example usage
    data_dir = '../data/raw'
    train_loader = get_data_loader(data_dir, 'train')
    
    # Print sample data
    for batch in train_loader:
        print(f"Batch size: {len(batch['file_id'])}")
        print(f"Waveform shape: {batch['waveform'].shape}")
        print(f"Sample rate: {batch['sample_rate'][0]}")
        print(f"Labels: {batch['label']}")
        break 