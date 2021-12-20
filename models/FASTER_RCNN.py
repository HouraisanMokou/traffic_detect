from models.roi.roi import ROI
from models.rpn.rpn import RPN
from models.rpn.proposal_target import RPN_PROPOSAL_TARGET
from models.bones.Resnet_50 import ResNet
from torch import nn
from torch import functional as F


class Faster_RCNN(nn.Module):
    """
    faster-RCNN structure
    """

    # def forward(self):
    def __init__(self, classes):
        self.classes = classes
        self.n_classes = len(classes)

        # base
        self.base = ResNet()

        # rpn
        self.rpn = RPN()
        self.proposal_target = RPN_PROPOSAL_TARGET()

        # roi
        self.roi_pool = ROI(7, 7, 1 / 16)
        self.linear1 = nn.Linear(512 * 7 * 7, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear_cls_prob = nn.Linear(4096, self.n_classes)
        self.linear_bbox_pred = nn.Linear(4096, self.n_classes * 4)

        # loss
        self.cross_entropy = None
        self.loss_box = None

    def loss(self):
        return self.cross_entropy + self.loss_box * 10

    def forward(self, feed_dicts):
        """
        :param feed_dict:
        :return:
        """
        batch_size = len(feed_dicts)
        im_info, gt_boxes, num_box = feed_dicts['im_info'], feed_dicts['gt_boxes'], feed_dicts['num_box']

        feats=self.base(im_info)
