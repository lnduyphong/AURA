import torch
import torch.nn as nn
import torch.nn.functional as F


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
