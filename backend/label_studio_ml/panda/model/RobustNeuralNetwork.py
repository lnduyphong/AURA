import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustNeuralNetwork(nn.Module):
    def __init__(self, input_size, n_labels):
        super(RobustNeuralNetwork, self).__init__()
        
        # Feature Layers with BatchNorm and increased dropout in deeper layers
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)  # Increased dropout
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(512, 512)  # Wider layer
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.6)
        
        # Residual connection to handle dimension mismatch
        self.residual1 = nn.Linear(512, 512)  # Identity if dimensions match
        
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.6)
        
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc6 = nn.Linear(128, n_labels)
        
        # Initialize weights using Kaiming initialization for GELU
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input block
        x = self.dropout1(F.gelu(self.bn1(self.fc1(x))))
        
        # First processing block
        x = self.dropout2(F.gelu(self.bn2(self.fc2(x))))
        
        # Residual block
        identity = x
        x = F.gelu(self.bn3(self.fc3(x)))
        x = self.residual1(identity) + x  # Skip connection
        x = self.dropout3(x)
        
        # Subsequent layers
        x = self.dropout4(F.gelu(self.bn4(self.fc4(x))))
        x = self.dropout5(F.gelu(self.bn5(self.fc5(x))))
        
        # Final output (logits)
        x = torch.softmax(self.fc6(x), dim=1)
        return x