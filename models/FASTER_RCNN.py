from models.roi.roi import ROI
from models.rpn.rpn import RPN
from models.bones.Resnet_50 import ResNet
from torch import nn
from torch import functional as F

class Faster_RCNN(nn.Module):
    """
    faster-RCNN structure
    """

    # def forward(self):
    def __init__(self,classes):
        self.classes=classes
        self.n_classes=len(classes)
