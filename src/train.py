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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

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

def prepare_features_batch(batch, feature_type='mel_spectrogram', device=None):
    """
    Prepare features batch for model input.
    
    Args:
        batch: Batch from dataloader
        feature_type: Type of feature to use
        device: Device to use
        
    Returns:
        dict: Prepared batch with features and labels
    """
    # Get features
    if 'features' in batch:
        # Use preprocessed features from SLUEDataset
        if isinstance(batch['features'], dict):
            # If features is a dictionary, use specified feature type
            if feature_type in batch['features']:
                features = batch['features'][feature_type]
            else:
                # Fallback to first available feature type
                feature_type = list(batch['features'].keys())[0]
                features = batch['features'][feature_type]
                print(f"Warning: Requested feature type '{feature_type}' not found. Using '{feature_type}' instead.")
        else:
            # If features is not a dictionary, use as is
            features = batch['features']
    elif 'waveform' in batch:
        # Extract features from waveform (not implemented, just placeholder)
        features = batch['waveform']
        print("Warning: Feature extraction from waveform not fully implemented.")
    else:
        raise ValueError("No features or waveform found in batch.")
    
    # If features is a 3D tensor (batch, freq, time), add channel dimension for CNN
    if len(features.shape) == 3:
        features = features.unsqueeze(1)  # (batch, 1, freq, time)
    
    # Transfer to device
    if device is not None:
        features = features.to(device)
        batch['label'] = batch['label'].to(device)
    
    # Add prepared features to batch
    batch['features'] = features
    
    return batch

def train_epoch(model, data_loader, optimizer, criterion, device, feature_type='mel_spectrogram'):
    """
    Train one epoch.
    
    Args:
        model: Model to train
        data_loader: Data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        feature_type: Type of feature to use
        
    Returns:
        float: Average loss
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        # Prepare batch
        batch = prepare_features_batch(batch, feature_type, device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch['features'])
        loss = criterion(outputs, batch['label'])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device, feature_type='mel_spectrogram', mode='val'):
    """
    Evaluate the model.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Device to use
        feature_type: Type of feature to use
        mode: Evaluation mode ('val' or 'test')
        
    Returns:
        tuple: Loss, accuracy, F1 score
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating ({mode})"):
            # Prepare batch
            batch = prepare_features_batch(batch, feature_type, device)
            
            # Forward pass
            outputs = model(batch['features'])
            loss = criterion(outputs, batch['label'])
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Record results
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
    
    # Calculate metrics
    loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return loss, accuracy, f1

def predict(model, data_loader, device, feature_type='mel_spectrogram'):
    """
    Make predictions with the model.
    
    Args:
        model: Model to use
        data_loader: Data loader
        device: Device to use
        feature_type: Type of feature to use
        
    Returns:
        tuple: True labels, predicted labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            # Prepare batch
            batch = prepare_features_batch(batch, feature_type, device)
            
            # Forward pass
            outputs = model(batch['features'])
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Record results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
    
    return all_labels, all_preds

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
    # Check unique classes in the data
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    print(f"Unique classes in test set: {unique_classes}")
    
    # Handle the case where there's only one class
    if len(unique_classes) == 1:
        single_class_idx = unique_classes[0]
        single_class_name = class_names[single_class_idx]
        
        # Create a simple report manually
        accuracy = np.mean(np.array(y_pred) == np.array(y_true))
        report = f"Only one class ({single_class_name}) present in the test set.\n"
        report += f"Accuracy: {accuracy:.4f}\n"
        report += f"All samples belong to class: {single_class_name}"
        
        # Save the report
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        # Print warning
        print(f"WARNING: Only one class ({single_class_name}) found in the test set!")
        print("This means your dataset is extremely imbalanced or incorrectly split.")
        print("Try creating a more balanced dataset for meaningful evaluation.")
    else:
        # For multiple classes, use the normal classification report
        try:
            # Generate classification report
            used_class_names = [class_names[i] for i in unique_classes]
            report = classification_report(y_true, y_pred, target_names=used_class_names, digits=4)
            
            # Save report
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
                f.write(report)
        except ValueError as e:
            # Fallback in case of errors
            print(f"Error creating classification report: {e}")
            accuracy = np.mean(np.array(y_pred) == np.array(y_true))
            with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Error generating full report: {e}")
    
    # Plot confusion matrix if there are multiple classes
    if len(unique_classes) > 1:
        plot_confusion_matrix(y_true, y_pred, 
                              [class_names[i] for i in unique_classes], 
                              output_dir)

def train_model(args):
    """
    Train a model for declarative question detection.
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    if args.use_processed_data:
        # Use processed data directory
        train_path = os.path.join(args.data_dir, 'train_metadata.json')
        if not os.path.exists(train_path):
            raise ValueError(f"Processed data not found at {train_path}")
            
        # Load datasets
        train_dataset = SLUEDataset(args.data_dir, 'train')
        test_dataset = SLUEDataset(args.data_dir, 'test')
        
        # Calculate class weights for weighted loss
        train_labels = [item['label'] for item in train_dataset.metadata.values()]
        class_counts = np.bincount(train_labels)
        total_samples = len(train_labels)
        class_weights = torch.tensor([total_samples / (len(class_counts) * count) for count in class_counts], 
                                     dtype=torch.float32, device=device)
        print(f"Class weights: {class_weights}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            collate_fn=custom_collate_fn
        )
        val_loader = None  # No validation set
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            collate_fn=custom_collate_fn
        )
        
        # Print dataset statistics
        print("Dataset statistics:")
        print(f"Train: {len(train_dataset)} samples")
        print(f"Test: {len(test_dataset)} samples")
        
        # Check label distribution
        train_label_counts = np.bincount([item['label'] for item in train_dataset.metadata.values()])
        test_label_counts = np.bincount([item['label'] for item in test_dataset.metadata.values()])
        print(f"Train label distribution: {train_label_counts}")
        print(f"Test label distribution: {test_label_counts}")
    else:
        # Use raw data from data_cleaning
        if not os.path.exists(args.csv_path):
            raise ValueError(f"CSV file not found at {args.csv_path}")
            
        # Create dataloaders
        train_loader = get_data_loader(args.audio_dir, args.csv_path, 'train', args.batch_size)
        val_loader = get_data_loader(args.audio_dir, args.csv_path, 'dev', args.batch_size)
        test_loader = get_data_loader(args.audio_dir, args.csv_path, 'test', args.batch_size)
        
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    
    # Create model
    if args.model_type == 'cnn':
        model = CNN(
            input_channels=1,
            num_classes=args.num_classes,
            hidden_size=args.hidden_size
        )
    elif args.model_type == 'transformer':
        model = TransformerModel(
            input_dim=80,  # Mel spectrogram features
            hidden_dim=args.hidden_size,
            num_classes=args.num_classes,
            num_layers=args.num_layers,
            nhead=args.nhead,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    # Train the model
    best_metric = 0
    best_model_state = None
    train_losses = []
    eval_metrics = []
    
    for epoch in range(args.num_epochs):
        # Train one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, feature_type=args.feature_type)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        if val_loader is not None:
            eval_loss, eval_acc, eval_f1 = evaluate(model, val_loader, criterion, device, feature_type=args.feature_type)
            print(f"Epoch {epoch+1}/{args.num_epochs} - "
                  f"Train loss: {train_loss:.4f}, "
                  f"Val loss: {eval_loss:.4f}, "
                  f"Val acc: {eval_acc:.4f}, "
                  f"Val F1: {eval_f1:.4f}")
            
            # Update learning rate scheduler
            scheduler.step(eval_loss)
            
            # Save best model
            if eval_f1 > best_metric:
                best_metric = eval_f1
                best_model_state = model.state_dict()
                
            # Record metrics
            eval_metrics.append({
                'loss': eval_loss,
                'accuracy': eval_acc,
                'f1': eval_f1
            })
        else:
            print(f"Epoch {epoch+1}/{args.num_epochs} - Train loss: {train_loss:.4f}")
            
            # Save the model after each epoch (no validation set)
            model_path = os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), model_path)
    
    # Load best model (if validation was used)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device, feature_type=args.feature_type)
    print(f"Test performance - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save training results
    results = {
        'train_losses': train_losses,
        'eval_metrics': eval_metrics,
        'test_metrics': {
            'loss': test_loss,
            'accuracy': test_acc,
            'f1': test_f1
        }
    }
    
    results_path = os.path.join(args.output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save classification report and confusion matrix
    y_true, y_pred = predict(model, test_loader, device)
    save_results(y_true, y_pred, ['non-qy^d', 'qy^d'], args.output_dir)
    
    return model

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