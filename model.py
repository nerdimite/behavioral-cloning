import torch
import torch.nn as nn
import torch.nn.functional as F

class NvidiaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 5, 2)
        self.conv2 = nn.Conv2d(24, 36, 5, 2)
        self.conv3 = nn.Conv2d(36, 48, 5, 2)
        self.conv4 = nn.Conv2d(48, 64, 3, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1)
        
        self.dense = nn.Sequential(nn.Linear(1152, 100),
                                   nn.ELU(),
                                   nn.Dropout(0.25),
                                   nn.Linear(100, 50),
                                   nn.ELU(),
                                   nn.Linear(50, 10),
                                   nn.ELU(),
                                   nn.Linear(10, 1))
    
    def forward(self, x):
        
        batch_size = x.size(0)
        
        # Convolutional Pass
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        
        # Linear Pass
        x = x.reshape(batch_size, -1)
        out = self.dense(x)
        
        return out