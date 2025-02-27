import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input channels: 3 (RGB), Output channels: 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # Downsample by 2
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output channels: 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output channels: 128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),  # 128 channels, 4x4 feature maps
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)           # 10 classes for CIFAR-10
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x