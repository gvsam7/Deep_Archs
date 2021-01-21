"""
Deep Architectures: 3, 4, 5 CNN, AlexNet, VGG16
"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.Layers import ConvLayer

conv = ConvLayer._add_layer
conv_2 = ConvLayer._add_layer2
conv_vgg = ConvLayer._VGG_add_layer
conv_vgg_2 = ConvLayer._VGG_add_layer2


class CNN_3_Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.conv1, input_size = conv(input_size[0], 32, 3, input_size)
        self.conv2, input_size = conv(input_size[0], 64, 3, input_size)
        self.conv3, input_size = conv(input_size[0], 128, 3, input_size)

        input_size_flattened = np.product(input_size)
        self.fc1 = nn.Linear(input_size_flattened, 512)
        self.fc2 = nn.Linear(512, 10)

    def convs(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_4_Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.conv1, input_size = conv(input_size[0], 32, 3, input_size)
        self.conv2, input_size = conv(input_size[0], 64, 3, input_size)
        self.conv3, input_size = conv(input_size[0], 128, 3, input_size)
        self.conv4, input_size = conv(input_size[0], 256, 3, input_size)

        input_size_flattened = np.product(input_size)
        self.fc1 = nn.Linear(input_size_flattened, 512)
        self.fc2 = nn.Linear(512, 10)

    def convs(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_5_Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.conv1, input_size = conv(input_size[0], 32, 3, input_size)
        self.conv2, input_size = conv(input_size[0], 64, 3, input_size)
        self.conv3, input_size = conv(input_size[0], 128, 3, input_size)
        self.conv4, input_size = conv(input_size[0], 256, 3, input_size)
        self.conv5, input_size = conv(input_size[0], 512, 3, input_size)

        input_size_flattened = np.product(input_size)
        self.fc1 = nn.Linear(input_size_flattened, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def convs(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.conv1, input_size = conv(input_size[0], 64, 3, input_size)
        self.conv2, input_size = conv(input_size[0], 192, 3, input_size)
        self.conv3, input_size = conv_2(input_size[0], 384, 3, input_size)
        self.conv4, input_size = conv_2(input_size[0], 256, 3, input_size)
        self.conv5, input_size = conv(input_size[0], 256, 3, input_size)

        input_size_flattened = np.product(input_size)
        self.dropout_1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size_flattened, 4096)
        self.dropout_2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def convs(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.flatten(start_dim=1)
        x = F.dropout(F.relu(self.fc1(x)))
        x = F.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class VGG16(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.conv1, input_size = conv_vgg(input_size[0], 64, 3, input_size)
        self.conv2, input_size = conv_vgg(input_size[0], 128, 3, input_size)
        self.conv3, input_size = conv_vgg_2(input_size[0], 256, 3, input_size)
        self.conv4, input_size = conv_vgg_2(input_size[0], 512, 3, input_size)
        self.conv5, input_size = conv_vgg_2(input_size[0], 512, 3, input_size)

        input_size_flattened = np.product(input_size)
        self.fc1 = nn.Linear(input_size_flattened, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def convs(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x