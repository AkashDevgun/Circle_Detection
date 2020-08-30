import torch.nn as nn


# Network For Images in Pretty Standard Way, 5 Series of Convolutions followed by Fully connected and Last Layer
class CircleNet(nn.Module):
    def __init__(self):
        super(CircleNet, self).__init__()
        self.Layer1 = nn.Sequential(nn.Conv2d(1, 32, 5), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.Layer2 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.Layer3 = nn.Sequential(nn.Conv2d(64, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.Layer4 = nn.Sequential(nn.Conv2d(128, 128, 3), nn.BatchNorm2d(128), nn.ReLU())
        self.Layer5 = nn.Sequential(nn.Conv2d(128, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(1764, 256), nn.ReLU(), nn.Linear(256, 16), nn.ReLU())
        self.last = nn.Linear(16, 3)

    def forward(self, x):
        x = self.Layer5(self.Layer4(self.Layer3(self.Layer2(self.Layer1(x)))))
        B, C, H, W = x.shape
        x = x.view(-1, C * H * W)
        return self.last(self.fc(x))
