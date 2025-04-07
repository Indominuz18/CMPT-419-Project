import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
import copy
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import time
import shutil
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
NUM_MELS = 128
NUM_MFCC = 40

# Enhanced Dataset class
class AudioFeatureDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        
        # Convert to tensors
        mel_spec = torch.tensor(feature['mel_spectrogram'], dtype=torch.float)
        mfcc = torch.tensor(feature['mfcc'], dtype=torch.float)
        prosody = torch.tensor(feature['prosody'], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Apply transforms if any
        if self.transform:
            mel_spec, mfcc, prosody = self.transform(mel_spec, mfcc, prosody)
            
        return {
            'mel_spectrogram': mel_spec,
            'mfcc': mfcc,
            'prosody': prosody,
            'label': label
        }

# Data transforms and augmentation
class FeatureDropout:
    def __init__(self, dropout_rate=0.1):
        self.dropout_rate = dropout_rate
        
    def __call__(self, mel, mfcc, prosody):
        # Apply random feature dropout
        if np.random.rand() < self.dropout_rate:
            mask = torch.FloatTensor(mel.shape).uniform_() > self.dropout_rate
            mel = mel * mask
        
        if np.random.rand() < self.dropout_rate:
            mask = torch.FloatTensor(mfcc.shape).uniform_() > self.dropout_rate
            mfcc = mfcc * mask
            
        if np.random.rand() < self.dropout_rate:
            mask = torch.FloatTensor(prosody.shape).uniform_() > self.dropout_rate
            prosody = prosody * mask
            
        return mel, mfcc, prosody

# Squeeze-and-Excitation block for channel attention
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Residual block with squeeze-and-excitation
class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(ResidualSEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

# Improved attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections and reshape
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scale dot-product attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(Q.device)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.num_heads * self.head_dim)
        x = self.fc_out(x)
        
        return x

# Enhanced CNN+LSTM model with advanced features
class EnhancedModel(nn.Module):
    def __init__(self, device=None):
        super(EnhancedModel, self).__init__()
        
        self.device = device if device is not None else torch.device('cpu')
        
        # CNN with residual blocks for Mel Spectrogram
        self.cnn_mel = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),  # Input channels = 2 for stacked mel features
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            ResidualSEBlock(32, 64, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            ResidualSEBlock(64, 128, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3)
        )
        
        # CNN for MFCC
        self.cnn_mfcc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            ResidualSEBlock(32, 64, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2)
        )
        
        # CNN for Prosody
        self.cnn_prosody = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2)
        )
        
        # Add dummy forward pass to calculate output dimensions
        self.lstm_mel_input_dim = None
        self.lstm_mfcc_input_dim = None
        self.lstm_prosody_input_dim = None
        
        # These LSTM layers will be properly initialized in the first forward pass
        self.lstm_mel = None
        self.lstm_mfcc = None
        self.lstm_prosody = None
        
        # Attention mechanisms
        self.attention_mel = None
        self.attention_mfcc = None
        self.attention_prosody = None
        
        # Fully connected layers - these dimensions will be set in the first forward pass
        self.fc = None
        
        # Layer normalization
        self.layer_norm_mel = None
        self.layer_norm_mfcc = None
        self.layer_norm_prosody = None
        
        # Move everything to the correct device
        self.to(self.device)
        
    def _initialize_lstms(self, mel_dim, mfcc_dim, prosody_dim):
        """Initialize LSTM layers with the correct input dimensions"""
        self.lstm_mel_input_dim = mel_dim
        self.lstm_mfcc_input_dim = mfcc_dim
        self.lstm_prosody_input_dim = prosody_dim
        
        # Hidden dimensions
        mel_hidden_dim = 128
        mfcc_hidden_dim = 64
        prosody_hidden_dim = 64
        
        # Initialize LSTM layers with correct dimensions and bidirectional
        self.lstm_mel = nn.LSTM(mel_dim, mel_hidden_dim, num_layers=2, 
                               batch_first=True, bidirectional=True, dropout=0.3)
        self.lstm_mfcc = nn.LSTM(mfcc_dim, mfcc_hidden_dim, num_layers=2, 
                                batch_first=True, bidirectional=True, dropout=0.3)
        self.lstm_prosody = nn.LSTM(prosody_dim, prosody_hidden_dim, num_layers=2, 
                                   batch_first=True, bidirectional=True, dropout=0.3)
        
        # Initialize attention mechanisms with correct dimensions
        self.attention_mel = MultiHeadAttention(mel_hidden_dim*2, num_heads=4, dropout=0.1)
        self.attention_mfcc = MultiHeadAttention(mfcc_hidden_dim*2, num_heads=4, dropout=0.1)
        self.attention_prosody = MultiHeadAttention(prosody_hidden_dim*2, num_heads=4, dropout=0.1)
        
        # Layer normalization
        self.layer_norm_mel = nn.LayerNorm(mel_hidden_dim*2)
        self.layer_norm_mfcc = nn.LayerNorm(mfcc_hidden_dim*2)
        self.layer_norm_prosody = nn.LayerNorm(prosody_hidden_dim*2)
        
        # Combined dimensions
        combined_dim = mel_hidden_dim*2 + mfcc_hidden_dim*2 + prosody_hidden_dim*2
        
        # Initialize fully connected layers with improved architecture
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 256),
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
        
        # Move newly created modules to the device
        if self.device:
            self.lstm_mel = self.lstm_mel.to(self.device)
            self.lstm_mfcc = self.lstm_mfcc.to(self.device)
            self.lstm_prosody = self.lstm_prosody.to(self.device)
            self.attention_mel = self.attention_mel.to(self.device)
            self.attention_mfcc = self.attention_mfcc.to(self.device)
            self.attention_prosody = self.attention_prosody.to(self.device)
            self.layer_norm_mel = self.layer_norm_mel.to(self.device)
            self.layer_norm_mfcc = self.layer_norm_mfcc.to(self.device)
            self.layer_norm_prosody = self.layer_norm_prosody.to(self.device)
            self.fc = self.fc.to(self.device)
    
    def to(self, device):
        """Override to method to ensure all components move to the right device"""
        self.device = device
        
        # Call the parent to method
        super(EnhancedModel, self).to(device)
        
        # Ensure all components are on the right device
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, nn.Module) and attr is not self:
                attr.to(device)
            
        return self
        
    def forward(self, mel_spec, mfcc, prosody):
        batch_size = mel_spec.size(0)
        
        # Process mel spectrogram - Input has 2 channels from enhanced features
        mel_spec = self.cnn_mel(mel_spec)
        mel_spec = mel_spec.permute(0, 2, 1, 3)  # Rearrange for LSTM [batch, seq_len, channels, features]
        mel_cnn_out_shape = mel_spec.shape
        mel_spec = mel_spec.reshape(batch_size, mel_spec.size(1), -1)  # Flatten for LSTM
        
        # Process MFCC
        mfcc = mfcc.unsqueeze(1)  # Add channel dimension
        mfcc = self.cnn_mfcc(mfcc)
        mfcc = mfcc.permute(0, 2, 1, 3)  # Rearrange for LSTM
        mfcc_cnn_out_shape = mfcc.shape
        mfcc = mfcc.reshape(batch_size, mfcc.size(1), -1)  # Flatten for LSTM
        
        # Process prosody
        prosody = prosody.unsqueeze(1)  # Add channel dimension
        prosody = self.cnn_prosody(prosody)
        prosody = prosody.permute(0, 2, 1, 3)  # Rearrange for LSTM
        prosody_cnn_out_shape = prosody.shape
        prosody = prosody.reshape(batch_size, prosody.size(1), -1)  # Flatten for LSTM
        
        # Initialize LSTM layers if this is the first forward pass
        if self.lstm_mel is None:
            mel_feature_dim = mel_spec.size(2)
            mfcc_feature_dim = mfcc.size(2)
            prosody_feature_dim = prosody.size(2)
            
            print(f"Initializing LSTM layers with dimensions:")
            print(f"  Mel CNN output shape: {mel_cnn_out_shape}, LSTM input dim: {mel_feature_dim}")
            print(f"  MFCC CNN output shape: {mfcc_cnn_out_shape}, LSTM input dim: {mfcc_feature_dim}")
            print(f"  Prosody CNN output shape: {prosody_cnn_out_shape}, LSTM input dim: {prosody_feature_dim}")
            
            self._initialize_lstms(mel_feature_dim, mfcc_feature_dim, prosody_feature_dim)
        
        # Pass through LSTM layers
        mel_out, _ = self.lstm_mel(mel_spec)
        mfcc_out, _ = self.lstm_mfcc(mfcc)
        prosody_out, _ = self.lstm_prosody(prosody)
        
        # Apply layer normalization
        mel_out = self.layer_norm_mel(mel_out)
        mfcc_out = self.layer_norm_mfcc(mfcc_out)
        prosody_out = self.layer_norm_prosody(prosody_out)
        
        # Apply attention mechanism
        mel_out = self.attention_mel(mel_out, mel_out, mel_out)
        mfcc_out = self.attention_mfcc(mfcc_out, mfcc_out, mfcc_out)
        prosody_out = self.attention_prosody(prosody_out, prosody_out, prosody_out)
        
        # Get sequence-level representations by mean pooling
        mel_out = torch.mean(mel_out, dim=1)
        mfcc_out = torch.mean(mfcc_out, dim=1)
        prosody_out = torch.mean(prosody_out, dim=1)
        
        # Concatenate all features
        combined = torch.cat((mel_out, mfcc_out, prosody_out), dim=1)
        
        # Final classification
        output = self.fc(combined)
        return output

# Enhanced training process
def train_model(data_path='processed_data/processed_features.pkl', 
                model_dir='models',
                num_epochs=20,
                batch_size=16,
                learning_rate=0.001,
                test_size=0.2,
                k_folds=1,  # Simplify to just one fold for now
                use_ensemble=False):  # Disable ensemble for simplicity
    """Train the model on preprocessed features with a simple train/test split"""
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load the preprocessed data
    print(f"Loading data from {data_path}...")
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Ensure data is properly loaded
        if not isinstance(data, dict) or 'features' not in data or 'labels' not in data:
            raise ValueError("Data format is incorrect. Expected dict with 'features' and 'labels' keys.")
            
        # Convert data to lists to ensure consistent handling
        features = list(data['features'])
        labels = list(data['labels'])
        filenames = list(data.get('filenames', []))
        
        # Ensure each data record has valid types
        valid_features = []
        valid_labels = []
        valid_filenames = []
        
        for i, (feature, label) in enumerate(zip(features, labels)):
            if isinstance(feature, dict) and 'mel_spectrogram' in feature and 'mfcc' in feature and 'prosody' in feature:
                valid_features.append(feature)
                valid_labels.append(int(label))  # Convert to int to ensure consistent type
                if i < len(filenames):
                    valid_filenames.append(filenames[i])
                else:
                    valid_filenames.append(f"sample_{i}")  # Add placeholder filenames if missing
        
        # Replace original data with validated data
        features = valid_features
        labels = valid_labels
        filenames = valid_filenames
        
        print(f"Loaded and validated {len(features)} samples")
        
        if len(features) == 0:
            raise ValueError("No valid features found in the dataset.")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return {
            'best_f1_score': 0.0,
            'error': str(e)
        }
    
    # Analyze class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    print(f"Class distribution: {label_counts}")
    
    # Calculate class weights based on inverse frequency
    if len(unique_labels) == 2:
        # Calculate inverse frequency
        statement_weight = len(labels) / (2 * counts[0]) if counts[0] > 0 else 1.0
        question_weight = len(labels) / (2 * counts[1]) if counts[1] > 0 else 1.0
        
        # Normalize weights
        total_weight = statement_weight + question_weight
        statement_weight = statement_weight / total_weight
        question_weight = question_weight / total_weight
        
        # Ensure question class has higher weight
        if statement_weight >= question_weight:
            question_weight = statement_weight * 2.0
        
        class_weights = [statement_weight, question_weight]
        print(f"Using class weights: {class_weights}")
    else:
        class_weights = None
        print("Using uniform class weights (no weighting)")
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # For tracking overall best model
    best_overall_model = None
    best_overall_score = 0.0
    
    # SIMPLIFIED APPROACH: Use a simple train/validation split
    # Use scikit-learn's train_test_split with stratification
    train_features, val_features, train_labels, val_labels = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    print(f"Training set size: {len(train_features)}, Validation set size: {len(val_features)}")
    print(f"Training class distribution: {np.bincount(train_labels)}")
    print(f"Validation class distribution: {np.bincount(val_labels)}")
    
    # Balance training data through oversampling
    statement_indices = [i for i, label in enumerate(train_labels) if label == 0]
    question_indices = [i for i, label in enumerate(train_labels) if label == 1]
    
    # Determine which class is the minority
    if len(statement_indices) > len(question_indices):
        minority_indices = question_indices
        majority_indices = statement_indices
        minority_label = 1
    else:
        minority_indices = statement_indices
        majority_indices = question_indices
        minority_label = 0
    
    # Oversample the minority class
    oversampled_indices = minority_indices * (len(majority_indices) // len(minority_indices))
    remaining = len(majority_indices) - len(oversampled_indices)
    if remaining > 0 and len(minority_indices) > 0:  # Avoid division by zero
        oversampled_indices += minority_indices[:remaining]
    
    # Combine majority and oversampled minority indices
    balanced_indices = majority_indices + oversampled_indices
    
    # Create balanced training set
    balanced_train_features = [train_features[i] for i in balanced_indices]
    balanced_train_labels = [train_labels[i] if i < len(train_labels) else minority_label for i in balanced_indices]
    
    print(f"Original training set size: {len(train_labels)}")
    print(f"Balanced training set size: {len(balanced_train_labels)}")
    print(f"Balanced class distribution: {np.bincount(balanced_train_labels)}")
    
    # Create datasets with transforms
    train_dataset = AudioFeatureDataset(balanced_train_features, balanced_train_labels, transform=FeatureDropout(dropout_rate=0.1))
    val_dataset = AudioFeatureDataset(val_features, val_labels)
    
    # Create data loaders with reduced number of workers for stability
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model with device
    model = EnhancedModel(device=device)
    
    # Define loss function with class weights
    if class_weights:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print(f"Using weighted loss with weights: {weight_tensor}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard cross entropy loss")
    
    # Define optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler - Cosine annealing with restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=learning_rate/100
    )
    
    # Initialize tracking variables
    best_val_f1 = 0.0
    best_model_state = None
    patience = 15  # Increased from 5 to 15 to allow more training epochs
    patience_counter = 0
    
    # For tracking metrics
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []
    
    # Training loop
    print("Starting training...")
    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_preds = []
            train_true = []
            
            # Progress bar for training
            train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch in train_loop:
                mel_spec = batch['mel_spectrogram'].to(device)
                mfcc = batch['mfcc'].to(device)
                prosody = batch['prosody'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(mel_spec, mfcc, prosody)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Track statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_true.extend(labels.cpu().numpy())
                
                # Update progress bar
                train_loop.set_postfix(loss=loss.item())
            
            # Update learning rate
            scheduler.step()
            
            # Calculate training metrics
            epoch_train_loss = train_loss / len(train_loader)
            train_f1 = f1_score(train_true, train_preds, average='macro')
            train_losses.append(epoch_train_loss)
            train_f1_scores.append(train_f1)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_true = []
            val_outputs = []
            
            # Progress bar for validation
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            with torch.no_grad():
                for batch in val_loop:
                    mel_spec = batch['mel_spectrogram'].to(device)
                    mfcc = batch['mfcc'].to(device)
                    prosody = batch['prosody'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(mel_spec, mfcc, prosody)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
                    val_outputs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())
                    
                    # Update progress bar
                    val_loop.set_postfix(loss=loss.item())
            
            # Calculate validation metrics
            epoch_val_loss = val_loss / len(val_loader)
            val_f1 = f1_score(val_true, val_preds, average='macro')
            val_losses.append(epoch_val_loss)
            val_f1_scores.append(val_f1)
            
            # Calculate ROC AUC for validation set if possible
            try:
                val_probs = [prob[1] for prob in val_outputs]
                val_auc = roc_auc_score(val_true, val_probs)
                auc_info = f", Val AUC: {val_auc:.4f}"
            except Exception:
                auc_info = ""
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {epoch_train_loss:.4f}, "
                  f"Train F1: {train_f1:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, "
                  f"Val F1: {val_f1:.4f}{auc_info}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(f"New best model with F1 Score: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        # If we have a best model state, we can still save it
        if best_model_state is None:
            return {
                'best_f1_score': 0.0,
                'error': str(e)
            }
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    
    # Save best model
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    
    # Save the model to disk
    torch.save({
        'model_state_dict': best_model_state if best_model_state is not None else model.state_dict(),
        'f1_score': best_val_f1,
        'lstm_dimensions': {
            'mel': model.lstm_mel_input_dim,
            'mfcc': model.lstm_mfcc_input_dim,
            'prosody': model.lstm_prosody_input_dim
        },
        'model_type': 'EnhancedModel',
        'timestamp': time.strftime("%Y%m%d-%H%M%S"),
        'class_weights': class_weights,
        'input_shapes': {
            'mel_spectrogram': (2, NUM_MELS, 432),
            'mfcc': (NUM_MFCC*3, 432),
            'prosody': (8, 432)
        }
    }, best_model_path)
    
    # Also save a dedicated copy for the demo application
    demo_model_path = os.path.join(model_dir, 'demo_model.pth')
    shutil.copy(best_model_path, demo_model_path)
    print(f"Demo-ready model saved to: {demo_model_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_f1_scores) + 1), train_f1_scores, label='Train F1')
    plt.plot(range(1, len(val_f1_scores) + 1), val_f1_scores, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_curves.png'))
    plt.close()
    
    # Save model architecture information
    model_info = {
        'input_shapes': {
            'mel_spectrogram': (2, NUM_MELS, 432),  # 2 channels
            'mfcc': (NUM_MFCC*3, 432),
            'prosody': (8, 432)  # 8 prosody features
        },
        'lstm_dimensions': {
            'mel': model.lstm_mel_input_dim,
            'mfcc': model.lstm_mfcc_input_dim,
            'prosody': model.lstm_prosody_input_dim
        },
        'parameters': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'best_f1_score': best_val_f1,
            'class_weights': class_weights
        }
    }
    
    with open(os.path.join(model_dir, 'model_info.pkl'), 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"\nTraining completed. Best model saved with F1 Score: {best_val_f1:.4f}")
    print(f"Model saved to: {best_model_path}")
    
    return {
        'best_f1_score': best_val_f1,
        'model_path': best_model_path,
        'demo_model_path': demo_model_path,
        'lstm_dimensions': {
            'mel': model.lstm_mel_input_dim,
            'mfcc': model.lstm_mfcc_input_dim,
            'prosody': model.lstm_prosody_input_dim
        }
    }

if __name__ == "__main__":
    results = train_model()
    print(f"\nTraining completed. Best model saved with F1 Score: {results['best_f1_score']:.4f}")
    print(f"Model saved to: {results['model_path']}")
    print(f"LSTM dimensions: {results['lstm_dimensions']}")
    if results['ensemble_size'] > 0:
        print(f"Ensemble of {results['ensemble_size']} models created") 