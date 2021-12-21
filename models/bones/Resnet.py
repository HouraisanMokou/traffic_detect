import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    """
    ResNet 50 and 101
    """
    def __init__(self, layer_sizes,num_class=1000):
        """
        :param layer_sizes:
        """
        self.inplane=64

        self.conv1=nn.Conv2d(3,self.inplane,kernel_size=7,stride=2)

