"""
cnn_model.py
============
PyTorch implementation of the ExoHabitNet 1D-CNN backbone.

It receives normalized, phase-folded Kepler flux light curves of shape (1, 1024)
and outputs logits for the 3 target classes:
- HABITABLE (0)
- NON_HABITABLE (1)
- FALSE_POSITIVE (2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ExoHabitNetCNN(nn.Module):
    def __init__(self, sequence_length=1024, num_classes=3):
        super(ExoHabitNetCNN, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Conv Block 2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Conv Block 3
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.4)
        
        # Global Average Pooling collapses the remaining time dimension
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output Classifier
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Input shape: (Batch Size, Channels, Sequence Length) -> (B, 1, 1024)
        """
        # Block 1: (B, 1, 1024) -> (B, 64, 512)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2: (B, 64, 512) -> (B, 128, 256)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3: (B, 128, 256) -> (B, 256, 256)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Global Pooling: (B, 256, 256) -> (B, 256, 1) -> (B, 256)
        x = self.adaptive_pool(x).squeeze(-1)
        
        # Output: (B, 256) -> (B, 3)
        logits = self.fc_out(x)
        # We don't use Softmax here because we plan to use nn.CrossEntropyLoss
        # which expects raw logits.
        
        return logits

def test_model():
    """
    Temporary script to ensure the model compiles and outputs correctly.
    """
    model = ExoHabitNetCNN()
    # Batch size: 32, Channels: 1, Sequence length: 1024
    dummy_input = torch.randn(32, 1, 1024)
    print(f"Input shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (32, 3), "Output shape is incorrect!"
    print("Model compiled and verified successfully!")

if __name__ == "__main__":
    test_model()
