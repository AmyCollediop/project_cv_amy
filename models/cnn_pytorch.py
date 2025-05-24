import torch
import torch.nn as nn
import torch.nn.functional as F

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        # Conv1: 3 canaux (RGB) -> 16 filtres, kernel 5x5
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        # Conv2: 16 -> 32 filtres, kernel 5x5
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2_drop = nn.Dropout2d(p=0.3)
        # Pooling: réduire la taille spatiale
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  
        self.fc1_drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 4)  

    def forward(self, x):
        # Conv1 + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv2 + BN + Dropout + ReLU + Pool
        x = self.pool(F.relu(self.bn2(self.conv2_drop(self.conv2(x)))))
        # Aplatir
        x = x.view(-1, 32 * 56 * 56)
        # FC1 + ReLU + Dropout
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        # FC2 + Softmax
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)