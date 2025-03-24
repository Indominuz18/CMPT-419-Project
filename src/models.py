"""
Module containing model architectures for speech understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """Simple CNN model for audio classification."""
    
    def __init__(self, input_channels=1, num_classes=10):
        """
        Initialize CNN model.
        
        Args:
            input_channels (int): Number of input channels
            num_classes (int): Number of output classes
        """
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        # Dynamically calculate the input size for the first linear layer
        # This would depend on the actual input size of the spectrograms
        # For example, assuming input is [batch_size, 1, 128, 400]:
        # After 4 pooling layers with kernel_size=2, it becomes [batch_size, 128, 8, 25]
        self.fc1 = nn.Linear(128 * 8 * 25, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Reshape for fully connected layer
        # Note: This assumes a fixed input size. In practice, you'd need to handle variable-sized inputs.
        try:
            x = x.view(x.size(0), -1)
        except:
            # If the input size doesn't match expectations, we'll use adaptive pooling to get a fixed size
            x = F.adaptive_avg_pool2d(x, (8, 25))
            x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class LSTMModel(nn.Module):
    """LSTM model for sequence classification."""
    
    def __init__(self, input_dim=40, hidden_dim=256, num_layers=2, num_classes=10, bidirectional=True):
        """
        Initialize LSTM model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden state dimension
            num_layers (int): Number of LSTM layers
            num_classes (int): Number of output classes
            bidirectional (bool): Whether to use bidirectional LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, input_dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step output
        last_output = lstm_out[:, -1, :]
        
        # Or use mean pooling
        # last_output = torch.mean(lstm_out, dim=1)
        
        # Fully connected layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output

class Wav2Vec2FineTuned(nn.Module):
    """Fine-tuned Wav2Vec2 model for speech classification."""
    
    def __init__(self, model_name="facebook/wav2vec2-base-960h", num_classes=10, freeze_feature_extractor=True):
        """
        Initialize fine-tuned Wav2Vec2 model.
        
        Args:
            model_name (str): Pretrained model name
            num_classes (int): Number of output classes
            freeze_feature_extractor (bool): Whether to freeze the feature extractor
        """
        super(Wav2Vec2FineTuned, self).__init__()
        
        from transformers import Wav2Vec2Model
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze feature extractor
        if freeze_feature_extractor:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
        
        # Get the output dimension of Wav2Vec2
        self.output_dim = self.wav2vec2.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_values):
        """
        Forward pass.
        
        Args:
            input_values (torch.Tensor): Input audio tensor
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        # Extract features with Wav2Vec2
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        
        # Global average pooling
        hidden_states = torch.mean(hidden_states, dim=1)
        
        # Classify
        logits = self.classifier(hidden_states)
        
        return logits

# Helper function to get model by name
def get_model(model_name, **kwargs):
    """
    Get model by name.
    
    Args:
        model_name (str): Model name ('cnn', 'lstm', or 'wav2vec2')
        **kwargs: Model parameters
        
    Returns:
        nn.Module: Model
    """
    model_map = {
        'cnn': CNNModel,
        'lstm': LSTMModel,
        'wav2vec2': Wav2Vec2FineTuned
    }
    
    if model_name not in model_map:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(model_map.keys())}")
    
    return model_map[model_name](**kwargs)

if __name__ == '__main__':
    # Example usage
    
    # CNN model
    cnn_model = CNNModel(input_channels=1, num_classes=5)
    cnn_input = torch.randn(2, 1, 128, 400)  # [batch_size, channels, height, width]
    cnn_output = cnn_model(cnn_input)
    print(f"CNN output shape: {cnn_output.shape}")
    
    # LSTM model
    lstm_model = LSTMModel(input_dim=40, hidden_dim=256, num_classes=5)
    lstm_input = torch.randn(2, 100, 40)  # [batch_size, sequence_length, input_dim]
    lstm_output = lstm_model(lstm_input)
    print(f"LSTM output shape: {lstm_output.shape}") 