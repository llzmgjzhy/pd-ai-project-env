import torch.nn as nn
import torch
import math
import math
import torchvision.models as models


class CustomNet(nn.Module):
    def __init__(self, in_hight, in_wide, num_classes=2):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        # 这里需要根据您的网络和数据调整
        self.fc = nn.Linear(
            32 * (in_hight // 4) * (in_wide // 4), num_classes
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平
        y = torch.softmax(self.fc(x), dim=1)
        return y

class CustomWinNet(nn.Module):  # 带窗口的网络
    def __init__(self, in_hight, in_wide, num_classes=2):
        super(CustomWinNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        # 这里需要根据您的网络和数据调整
        self.fc = nn.Linear(
            32 * (in_hight // 4) * (in_wide // 4), num_classes
        )

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平
        y = torch.softmax(self.fc(x), dim=1)
        return y
