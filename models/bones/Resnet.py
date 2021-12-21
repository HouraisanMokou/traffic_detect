import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    """
    ResNet 50 and 101
    """

    def __init__(self, layer_sizes, num_class=1000):
        """
        :param layer_sizes:
        """
        self.inplane = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(self.inplane)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplane, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion)
        ) if stride != 1 or self.inplane != planes * block.expansion else None

        layers=[]
        layers.append(block(self.inplane,planes*block.expansion,stride,downsample))
        self.inplane=planes*block.expansion
        for l in range(blocks-1):
            layers.append(block(self.inplane,planes))

        return nn.Sequential(*layers)


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
