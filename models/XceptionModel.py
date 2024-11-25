import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class mini_XCEPTION(nn.Module):
    def __init__(self, input_channels, num_classes, l2_regularization=0.01):
        super(mini_XCEPTION, self).__init__()
        regularization = l2_regularization

        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(8)

        # Module 1
        self.residual1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(16)
        )
        self.sepconv1 = SeparableConv2d(8, 16, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.sepconv2 = SeparableConv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Module 2
        self.residual2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(32)
        )
        self.sepconv3 = SeparableConv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.sepconv4 = SeparableConv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Module 3
        self.residual3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64)
        )
        self.sepconv5 = SeparableConv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.sepconv6 = SeparableConv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Module 4
        self.residual4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128)
        )
        self.sepconv7 = SeparableConv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(128)
        self.sepconv8 = SeparableConv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(128, num_classes, kernel_size=3, padding=1, bias=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # Module 1
        residual = self.residual1(x)
        x = self.relu(self.bn3(self.sepconv1(x)))
        x = self.bn4(self.sepconv2(x))
        x = self.maxpool1(x) + residual

        # Module 2
        residual = self.residual2(x)
        x = self.relu(self.bn5(self.sepconv3(x)))
        x = self.bn6(self.sepconv4(x))
        x = self.maxpool2(x) + residual

        # Module 3
        residual = self.residual3(x)
        x = self.relu(self.bn7(self.sepconv5(x)))
        x = self.bn8(self.sepconv6(x))
        x = self.maxpool3(x) + residual

        # Module 4
        residual = self.residual4(x)
        x = self.relu(self.bn9(self.sepconv7(x)))
        x = self.bn10(self.sepconv8(x))
        x = self.maxpool4(x) + residual

        x = self.conv3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    input_channels = 1  # Grayscale images
    num_classes = 7  # Number of classes for facial expressions
    model = mini_XCEPTION(input_channels, num_classes)
    print(model)
