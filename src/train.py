"""
Module for training models on the SLUE dataset.
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

from data_loader import get_data_loader
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
        self.file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.pt')]
        
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
            'features': data['features'],
            'label': data['label']
        }

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        float: Training loss
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        
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

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        tuple: (evaluation loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
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
    # Create classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Save to JSON
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, output_dir)

def main(args):
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    
    # Set up data loaders
    if args.use_processed_data:
        # Use preprocessed features
        train_dataset = ProcessedDataset(args.data_dir, 'train')
        dev_dataset = ProcessedDataset(args.data_dir, 'dev')
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    else:
        # Use raw data and extract features on-the-fly
        train_loader = get_data_loader(
            args.data_dir,
            'train',
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        dev_loader = get_data_loader(
            args.data_dir,
            'dev',
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    
    # Create model
    model = get_model(args.model_type, num_classes=args.num_classes)
    model = model.to(device)
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # Training loop
    best_accuracy = 0.0
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_time = time.time() - start_time
        
        # Evaluate
        eval_loss, eval_acc, _, _ = evaluate(model, dev_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(eval_loss)
        
        # Print results
        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Time: {train_time:.2f}s")
        print(f"Eval loss: {eval_loss:.4f}, Eval acc: {eval_acc:.4f}")
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/eval', eval_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/eval', eval_acc, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if eval_acc > best_accuracy:
            best_accuracy = eval_acc
            torch.save(model.state_dict(), os.path.join(args.model_dir, f'best_model_{args.model_type}.pt'))
            print(f"Saved best model with accuracy {best_accuracy:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'train_acc': train_acc,
                'eval_acc': eval_acc
            }, os.path.join(args.model_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Final evaluation
    print("\nFinal evaluation:")
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(args.model_dir, f'best_model_{args.model_type}.pt')))
    
    # Evaluate on dev set
    _, _, dev_preds, dev_labels = evaluate(model, dev_loader, criterion, device)
    
    # Save results
    save_results(
        dev_labels,
        dev_preds,
        class_names=[str(i) for i in range(args.num_classes)],
        output_dir=args.output_dir
    )
    
    print(f"Training complete. Best accuracy: {best_accuracy:.4f}")
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models on SLUE dataset')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing data')
    parser.add_argument('--use_processed_data', action='store_true',
                        help='Use preprocessed features')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='cnn',
                        choices=['cnn', 'lstm', 'wav2vec2'],
                        help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of output classes')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoints every N epochs')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='Directory to save results')
    parser.add_argument('--model_dir', type=str, default='../models',
                        help='Directory to save models')
    
    args = parser.parse_args()
    main(args) 