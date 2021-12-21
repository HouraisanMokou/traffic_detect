from models.roi.roi import ROI
from models.rpn.rpn import RPN
from models.rpn.proposal_target import RPN_PROPOSAL_TARGET
from models.bones.Resnet import ResNet
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class Faster_RCNN(nn.Module):
    """
    faster-RCNN structure
    """

    # def forward(self):
    def __init__(self, classes, args):
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

        # args
        self.dropout_rate = args.dropout_rate

    def loss(self):
        return self.cross_entropy + self.loss_box * 10

    def forward(self, feed_dicts):
        """
        :param feed_dict:
        :return:
        """
        batch_size = len(feed_dicts)
        im_data, im_info, gt_boxes, num_boxes = \
            feed_dicts['im_data'], feed_dicts['im_info'], feed_dicts['gt_boxes'], feed_dicts['num_box']

        feats = self.base(im_data, im_info, gt_boxes, num_boxes)

        rois = self.rpn(feats)

        if self.training:
            rois_data = self.proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, roi_inside_ws, roi_outside_ws = rois_data

        # roi pool
        feats_pooled = self.roi_pool(feats, rois)
        feats_linear = feats_pooled.view(feats_pooled.size()[0], -1)
        feats_linear = F.dropout(F.relu(self.linear1(feats_linear)), self.dropout_rate, training=self.training)
        feats_linear = F.dropout(F.relu(self.linear2(feats_linear)), self.dropout_rate, training=self.training)

        cls_prob = F.softmax(self.linear_cls_prob(feats_linear))
        bbox_pred = self.linear_bbox_pred(feats_linear)

        if self.training:
            self.cross_entropy, self.loss_box = self.build_loss(cls_prob, bbox_pred, rois_data)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return cls_prob, bbox_pred, rois
