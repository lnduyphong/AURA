import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustNeuralNetwork(nn.Module):
    def __init__(self, input_size, n_labels):
        super(RobustNeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.6)
        
        self.residual1 = nn.Linear(512, 512) 
        
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.6)
        
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc6 = nn.Linear(128, n_labels)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.bn1(self.fc1(x))))
        
        x = self.dropout2(F.gelu(self.bn2(self.fc2(x))))
        
        identity = x
        x = F.gelu(self.bn3(self.fc3(x)))
        x = self.residual1(identity) + x
        x = self.dropout3(x)
        
        x = self.dropout4(F.gelu(self.bn4(self.fc4(x))))
        x = self.dropout5(F.gelu(self.bn5(self.fc5(x))))
        
        x = torch.softmax(self.fc6(x), dim=1)
        return x


class LeakyNeuralNetwork(nn.Module):
    def __init__(self, input_size, n_labels):
        super(LeakyNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(256, 128)
        self.dropout4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(128, 32)
        self.dropout5 = nn.Dropout(0.3)
        self.fc6 = nn.Linear(32, n_labels)
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.leaky_relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.leaky_relu(self.fc4(x))
        x = self.dropout4(x)
        x = self.leaky_relu(self.fc5(x))
        x = self.dropout5(x)
        x = torch.softmax(self.fc6(x), dim=1)
        return x
