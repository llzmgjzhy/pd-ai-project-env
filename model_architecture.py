import torch.nn as nn
import torch
import math
import math
import torchvision.models as models

class CustomNet(nn.Module):
    def __init__(self, in_height, in_width, num_classes=2):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.fc1 = nn.Linear(32 * (in_height // 4) * (in_width // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.functional.relu(self.fc1(x))
        y = self.fc2(x)
        # y = F.softmax(y, dim=1)
        return y
# class CustomNet(nn.Module):
#     def __init__(self, in_hight, in_wide, num_classes=1):
#         super(CustomNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
#         self.act1 = nn.ReLU()
#         self.pool = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.act2 = nn.ReLU()
#         # 这里需要根据您的网络和数据调整
#         self.fc = nn.Linear(
#             32 * (in_hight // 4) * (in_wide // 4), num_classes
#         )

#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.pool(self.act1(self.conv1(x)))
#         x = self.pool(self.act2(self.conv2(x)))
#         x = x.view(x.size(0), -1)  # 展平
#         # y = torch.softmax(self.fc(x), dim=1)
#         y=torch.sigmoid(self.fc(x))
#         return y


class CustomWinNet(nn.Module):  # 带窗口的网络
    def __init__(self, in_height, in_width, num_classes=2):
        super(CustomWinNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.fc1 = nn.Linear(32 * (in_height // 4) * (in_width // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.functional.relu(self.fc1(x))
        y = self.fc2(x)
        return y
    
class VoltageNet(nn.Module):
    def __init__(self, input_dim=4, output_dim=1):
        super(VoltageNet, self).__init__()
        # 定义输入层到隐藏层的全连接层
        self.fc1 = nn.Linear(4, 8)  # 输入特征4，隐藏层神经元8
        # 定义隐藏层到输出层的全连接层
        self.fc2 = nn.Linear(8, 1)  # 隐藏层神经元8，输出标签1

    def forward(self, x):
        # 前向传播
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数作为隐藏层的激活函数
        x = torch.sigmoid(self.fc2(x))  # 使用Sigmoid激活函数作为输出层的激活函数，将输出限制在0到1之间
        return x


class VoltageWinNet(nn.Module):
    def __init__(self, input_dim=4, output_dim=1, hidden_dim1=8, hidden_dim2=4):
        super(VoltageWinNet, self).__init__()
        # 第一层：输入层到隐藏层1
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        # 第二层：隐藏层1到隐藏层2
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # 第三层：隐藏层2到输出层
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # 前向传播
        batch_size, window_size, input_dim = x.shape
        # 合并batch_size和window_size，以适应全连接层
        x = x.view(batch_size * window_size, input_dim)

        # 通过第一层
        x = torch.relu(self.fc1(x))
        # 通过第二层
        x = torch.relu(self.fc2(x))
        # 通过第三层
        x = torch.sigmoid(self.fc3(x))
        # 恢复原始维度[batch_size, window_size, output_dim]
        x = x.view(batch_size, window_size, -1)
        # 在window_size维度上取平均，输出维度为[batch_size, output_dim]
        return torch.mean(x, dim=1).view(-1)