import torch
import torch.nn as nn
import torch.nn.functional as F

def hflip(x):
    return torch.flip(x, dims=[-1])

class ParityInvariantBlock(nn.Module):
    """
    phi(x) = 0.5 * (Conv(x) + mirror_back(Conv(mirror(x)))).
    Enforces layer-wise invariance to horizontal reflection (Z2 parity).
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)

    def forward(self, x):
        y1 = self.conv(x)
        x_m = hflip(x)
        y2 = self.conv(x_m)
        y2 = hflip(y2)  # map back
        return 0.5 * (y1 + y2)

class ParityInvariantCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.block1 = ParityInvariantBlock(1, 16, 3, 1)
        self.block2 = ParityInvariantBlock(16, 32, 3, 1)
        self.block3 = ParityInvariantBlock(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*4*4, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.block1(x))
        x = self.pool(F.relu(self.block2(x)))  # 16x16
        x = self.pool(F.relu(self.block3(x)))  # 8x8 -> 4x4
        x = self.pool(x)                       # 4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
