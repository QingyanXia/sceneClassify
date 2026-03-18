import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 模块"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale


class ResidualBlockSE(nn.Module):
    """带SE模块的残差块"""
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(ResidualBlockSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # 在残差连接前添加SE
        out += residual
        return F.relu(out)


class SEResNet(nn.Module):
    """带SE模块的简化ResNet（类似ResNet-18结构）"""
    def __init__(self, num_classes=6, reduction=16):
        super(SEResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            ResidualBlockSE(64, 64, reduction=reduction),
            ResidualBlockSE(64, 64, reduction=reduction)
        )
        self.layer2 = nn.Sequential(
            ResidualBlockSE(64, 128, stride=2, reduction=reduction),
            ResidualBlockSE(128, 128, reduction=reduction)
        )
        self.layer3 = nn.Sequential(
            ResidualBlockSE(128, 256, stride=2, reduction=reduction),
            ResidualBlockSE(256, 256, reduction=reduction)
        )
        self.layer4 = nn.Sequential(
            ResidualBlockSE(256, 512, stride=2, reduction=reduction),
            ResidualBlockSE(512, 512, reduction=reduction)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x