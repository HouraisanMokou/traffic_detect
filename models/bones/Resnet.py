import math

import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    """
    ResNet 50 and 101
    """

    @staticmethod
    def ResNet50(n_classes):
        return ResNet([3, 4, 6, 3], num_class=n_classes)

    @staticmethod
    def ResNet101(n_classes):
        return ResNet([3, 4, 23, 3], num_class=n_classes)

    def __init__(self, layer_sizes, num_class=1000):
        """
        :param layer_sizes:
        """
        self.inplane = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(self.inplane)

        self.layer1 = self._make_layer(block, 64, layer_sizes[0])
        self.layer2 = self._make_layer(block, 128, layer_sizes[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer_sizes[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer_sizes[3], stride=2)

        self.linear = nn.Linear(512 * block.expansion, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplane, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion)
        ) if stride != 1 or self.inplane != planes * block.expansion else None

        layers = []
        layers.append(block(self.inplane, planes * block.expansion, stride, downsample))
        self.inplane = planes * block.expansion
        for l in range(blocks - 1):
            layers.append(block(self.inplane, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)), 7, 2, 3)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x=F.avg_pool2d(x,7)
        x=x.view(x.size()[0],-1)
        x=self.linear(x)
        return x


class block(nn.Module):
    expansion = 4

    def __init__(self, inplane, midplane, stride=1, downsample=None):
        super(block, self).__init__()

        self.conv1 = nn.Conv2d(inplane, midplane, stride)
        self.bn1 = nn.BatchNorm2d(midplane)

        self.conv2 = nn.Conv2d(midplane, midplane, stride)
        self.bn2 = nn.BatchNorm2d(midplane)

        self.conv3 = nn.Conv2d(midplane, midplane * self.expansion, stride)
        self.bn3 = nn.BatchNorm2d(midplane)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + res
        out = F.relu(out)

        return out
