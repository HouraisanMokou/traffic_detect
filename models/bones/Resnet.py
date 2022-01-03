import torch
import torch.nn as nn
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        print(x.size(),torch.cuda.memory_allocated(device='cuda:1')/(1024**3))
        output = self.conv1(x)
        print(output.size(),torch.cuda.memory_allocated(device='cuda:1')/(1024**3))
        output = self.conv2_x(output)
        print(output.size(),torch.cuda.memory_allocated(device='cuda:1')/(1024**3))
        output = self.conv3_x(output)
        print(output.size(),torch.cuda.memory_allocated(device='cuda:1')/(1024**3))

        return output

    def head_to_tail(self,x):
        output = self.conv4_x(x)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    @staticmethod
    def ResNet50(num_classes=100):
        """ return a ResNet 50 object
        """
        return ResNet(BottleNeck, [3, 4, 6, 3],num_classes=num_classes)


def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])
# import math
#
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class ResNet(nn.Module):
#     """
#     ResNet 50 and 101
#     """
#
#     @staticmethod
#     def ResNet50(n_classes):
#         return ResNet([3, 4, 6, 3], num_class=n_classes)
#
#     @staticmethod
#     def ResNet101(n_classes):
#         return ResNet([3, 4, 23, 3], num_class=n_classes)
#
#     def __init__(self, layer_sizes, num_class=1000):
#         """
#         :param layer_sizes:
#         """
#         self.inplane = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, stride=2)
#         self.bn1 = nn.BatchNorm2d(self.inplane)
#
#         self.layer1 = self._make_layer(block, 64, layer_sizes[0])
#         print(self.layer1)
#         self.layer2 = self._make_layer(block, 128, layer_sizes[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layer_sizes[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layer_sizes[3], stride=2)
#
#         self.linear = nn.Linear(512 * block.expansion, num_class)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         strides = [stride] + [1] * (blocks - 1)
#         layers = []
#         layers.append(block(self.inplane, planes * block.expansion, stride))
#         self.inplane = planes * block.expansion
#         for stride in strides:
#             layers.append(block(self.inplane, planes, stride))
#             self.inplane = planes * block.expansion
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.max_pool2d(F.relu(self.bn1(x)), 7, 2, 3)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x=F.avg_pool2d(x,7)
#         return x
#
#     def head_ti_tail(self,x):
#         x=x.view(x.size()[0],-1)
#         x=self.linear(x)
#         return x
#
#
# class block(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplane, midplane, stride=1):
#         super(block, self).__init__()
#
#         self.conv1 = nn.Conv2d(inplane, midplane, kernel_size=1,stride=stride)
#         self.bn1 = nn.BatchNorm2d(midplane)
#
#         self.conv2 = nn.Conv2d(midplane, midplane,kernel_size=3,stride=stride)
#         self.bn2 = nn.BatchNorm2d(midplane)
#
#         self.conv3 = nn.Conv2d(midplane, midplane * self.expansion, kernel_size=1,stride=stride)
#         self.bn3 = nn.BatchNorm2d(midplane * self.expansion)
#
#         self.downsample = nn.Sequential()
#
#         if stride != 1 or inplane != midplane * block.expansion:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(inplane, midplane * block.expansion, stride=stride, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(midplane * block.expansion)
#             )
#         self.stride = stride
#
#     def forward(self, x):
#         print(x.size())
#         res=self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = F.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = F.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         print(out.size(),res.size())
#         out = out + res
#         out = F.relu(out)
#
#         return out
