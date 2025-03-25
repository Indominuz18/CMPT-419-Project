"""
Module for training models on the declarative question detection dataset.
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import get_data_loader, DeclarativeQuestionDataset, SLUEDataset, custom_collate_fn
from models import get_model

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

class ProcessedDataset(Dataset):
    """Dataset class for preprocessed features."""
    
    def __init__(self, data_dir, split='train'):
        """
        Initialize dataset.
        
        Args:
            data_dir (str): Directory containing processed features
            split (str): Data split ('train', 'dev', or 'test')
        """
        self.data_dir = os.path.join(data_dir, split)
        
        # Check if directory exists
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory {self.data_dir} does not exist")
            
        self.file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.pt')]
        
        # Load metadata if available
        metadata_path = os.path.join(data_dir, f"{split}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None
    
    def __len__(self):
        """Return the number of samples in dataset."""
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: A dictionary containing the features and labels
        """
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(file_path)
        
        return {
            'file_id': data.get('file_id', self.file_list[idx]),
            'features': data['features'],
            'label': data['label']
        }

def prepare_features_batch(batch, feature_type='mel_spectrogram', device='cpu'):
    """
    Prepare features batch for model input.
    
    Args:
        batch: Batch of samples from DataLoader
        feature_type: Type of feature to use
        device: Device to use
        
    Returns:
        torch.Tensor: Batch of features
        torch.Tensor: Batch of labels
    """
    if isinstance(batch['features'], dict) and feature_type in batch['features']:
        # Features are stored in a dictionary
        features = batch['features'][feature_type].to(device)
    elif isinstance(batch['features'][0], dict) and feature_type in batch['features'][0]:
        # Each sample has a dictionary of features
        features = torch.stack([sample[feature_type] for sample in batch['features']]).to(device)
    else:
        # Features are already in the right format
        features = batch['features'].to(device)
    
    labels = batch['label'].to(device)
    
    return features, labels

def train_epoch(model, dataloader, criterion, optimizer, device, feature_type='mel_spectrogram'):
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use ('cuda' or 'cpu')
        feature_type: Type of feature to use
        
    Returns:
        float: Training loss
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        features, labels = prepare_features_batch(batch, feature_type, device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': total_loss / (progress_bar.n + 1),
            'acc': 100 * correct / total
        })
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device, feature_type='mel_spectrogram'):
    """
    Evaluate model.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to use ('cuda' or 'cpu')
        feature_type: Type of feature to use
        
    Returns:
        tuple: (evaluation loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            features, labels = prepare_features_batch(batch, feature_type, device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Save predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return total_loss / len(dataloader), accuracy, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def save_results(y_true, y_pred, class_names, output_dir):
    """
    Save classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save results
    """
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, output_dir)

def train_model(args):
    """
    Train a model.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set class names
    class_names = args.class_names.split(',')
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    
    # Set up TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    
    # Set up data loaders
    print("Loading data...")
    
    if args.use_processed_data:
        # Use processed data
        train_dataset = ProcessedDataset(args.data_dir, 'train')
        val_dataset = ProcessedDataset(args.data_dir, 'dev')
        test_dataset = ProcessedDataset(args.data_dir, 'test')
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
    else:
        # Use DeclarativeQuestionDataset
        if args.use_slue_dataset:
            # Use original SLUE dataset (if available)
            train_loader = get_data_loader(args.data_dir, 'train', args.batch_size, args.num_workers)
            val_loader = get_data_loader(args.data_dir, 'dev', args.batch_size, args.num_workers)
            test_loader = get_data_loader(args.data_dir, 'test', args.batch_size, args.num_workers)
        else:
            # Use DeclarativeQuestionDataset directly
            label_map = {class_name: i for i, class_name in enumerate(class_names)}
            train_loader = get_data_loader(
                args.audio_dir, 
                args.csv_path, 
                'train', 
                args.batch_size, 
                args.num_workers,
                label_map=label_map
            )
            val_loader = get_data_loader(
                args.audio_dir, 
                args.csv_path, 
                'dev', 
                args.batch_size, 
                args.num_workers,
                label_map=label_map
            )
            test_loader = get_data_loader(
                args.audio_dir, 
                args.csv_path, 
                'test', 
                args.batch_size, 
                args.num_workers,
                label_map=label_map
            )
    
    # Create model
    print(f"Creating {args.model_type} model...")
    model = get_model(
        args.model_type,
        input_channels=1,
        num_classes=num_classes
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    print("Training model...")
    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, feature_type=args.feature_type
        )
        
        # Evaluate
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device, feature_type=args.feature_type
        )
        
        # Save model if it's the best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation accuracy: {val_acc:.4f}")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Time: {time.time() - start_time:.2f}s")
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate on test set
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, feature_type=args.feature_type
    )
    
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    # Save results
    save_results(test_labels, test_preds, class_names, args.output_dir)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))
    
    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for declarative question detection")
    
    # Data options
    parser.add_argument("--data_dir", type=str, default="../data/processed",
                        help="Directory containing processed data")
    parser.add_argument("--audio_dir", type=str, default="../data_cleaning/split_audiov2",
                        help="Directory containing audio files")
    parser.add_argument("--csv_path", type=str, default="../data_cleaning/swda_declarative.csv",
                        help="Path to the swda_declarative.csv file")
    parser.add_argument("--use_processed_data", action="store_true",
                        help="Use processed data instead of audio files")
    parser.add_argument("--use_slue_dataset", action="store_true",
                        help="Use original SLUE dataset")
    parser.add_argument("--feature_type", type=str, default="mel_spectrogram",
                        help="Type of feature to use (mel_spectrogram, prosody, or wav2vec2)")
    
    # Model options
    parser.add_argument("--model_type", type=str, default="cnn",
                        help="Model type (cnn, lstm, or wav2vec2)")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes")
    parser.add_argument("--class_names", type=str, default="non-qy^d,qy^d",
                        help="Comma-separated list of class names")
    
    # Training options
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="../models/output",
                        help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    train_model(args) 