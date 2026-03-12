import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 输入3通道，输出32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2最大池化，步长2

        # 全连接层
        # 输入尺寸150x150，经过3次池化后：150/2/2/2 = 18.75 -> 18
        self.fc1 = nn.Linear(128 * 18 * 18, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 卷积 -> ReLU -> 池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # 展平
        x = x.view(x.size(0), -1)
        # 全连接
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x