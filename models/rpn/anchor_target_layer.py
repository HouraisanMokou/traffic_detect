from numpy.core.numeric import base_repr
import torch
from torch import random
import torch.nn as nn
from models.rpn.anchors import generate_anchors
import numpy as np
from models.rpn.bbox_transform import bbox_overlaps_batch, bbox_transform_batch
# import config as cfg


class anchor_target_layer(nn.Module):
    def __init__(self, feat_stride, scales, ratios,args) -> None:
        super(anchor_target_layer,self).__init__()

        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(
            scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.shape[0]

        self._allowed_border = 0
        self.args=args

    def forward(self, input):

        rpn_cls_score = input[0]
        gt_boxes = input[1]
        im_info = input[2]
        num_boxes = input[3]

        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        batch_size = gt_boxes.size(0)
        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        shift_x = np.arange(0, width)*self._feat_stride
        shift_y = np.arange(0, height)*self._feat_stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                            shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(gt_boxes)
        all_anchors = self._anchors.view(1, A, 4)+shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K*A, 4)

        total_anchors = int(K*A)

        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < int(im_info[0][1])+self._allowed_border) &
                (all_anchors[:, 3] < int(im_info[0][0])+self._allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)

        anchors = all_anchors[inds_inside, :]

        # label: 1 = positive, 0 = negative, -1 = don't care
        labels = gt_boxes.new(batch_size, inds_inside.shape[0]).fill_(-1)
        bbox_inside_weights = gt_boxes.new(
            batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(
            batch_size, inds_inside.size(0)).zero_()

        overlaps = bbox_overlaps_batch(anchors, gt_boxes)

        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        if not self.args.TRAIN_RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < self.args.TRAIN_RPN_NEGATIVE_OVERLAP] = 0

        gt_max_overlaps[gt_max_overlaps == 0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(
            batch_size, 1, -1).expand_as(overlaps)), 2)

        if torch.sum(keep) > 0:
            labels[keep > 0] = 1

        labels[max_overlaps >= self.args.TRAIN_RPN_POSITIVE_OVERLAP] = 1

        if self.args.TRAIN_RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < self.args.TRAIN_RPN_NEGATIVE_OVERLAP] = 0

        num_fg = int(self.args.TRAIN_RPN_FG_FRACTION*self.args.TRAIN_RPN_BATCHSIZE)

        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        for i in range(batch_size):
            # subsample positive labels if too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(
                    fg_inds.size(0))).type_as(gt_boxes).int()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

            num_bg = self.args.TRAIN_RPN_BATCHSIZE - \
                torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if too many
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)

                rand_num = torch.from_numpy(np.random.permutation(
                    bg_inds.size(0))).type_as(gt_boxes).int()
                disable_inds = bg_inds[rand_num[:bg_inds.shape[0]-num_bg]]
                labels[i][disable_inds] = -1

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)

        argmax_overlaps = argmax_overlaps + \
            offset.view(batch_size, 1).type_as(argmax_overlaps)
        bbox_targets = _compute_targets_batch(
            anchors, gt_boxes.view(-1, 5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))
        RPN_BBOX_INSIDE_WEIGHTS=(1.0, 1.0, 1.0, 1.0)
        bbox_inside_weights[labels == 1] = RPN_BBOX_INSIDE_WEIGHTS[0]

        if self.args.TRAIN_RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0/num_examples.item()
            negative_weights = 1.0/num_examples.item()
        else:
            assert((self.args.TRAIN_RPN_POSITIVE_WEIGHT > 0) &
                   (self.args.TRAIN_RPN_POSITIVE_WEIGHT < 1))

        bbox_outside_weights[labels==1]=positive_weights
        bbox_outside_weights[labels==0]=negative_weights

        labels=_unmap(labels,total_anchors,inds_inside,batch_size,fill=-1)
        bbox_targets=_unmap(bbox_targets,total_anchors,inds_inside,batch_size,fill=0)
        bbox_inside_weights=_unmap(bbox_inside_weights,total_anchors,inds_inside,batch_size,fill=0)
        bbox_outside_weights=_unmap(bbox_outside_weights,total_anchors,inds_inside,batch_size,fill=0)

        outputs=[]

        labels=labels.view(batch_size,height,width,A).permute(0,3,1,2).contiguous()
        labels=labels.view(batch_size,1,A*height,width)
        outputs.append(labels)

        bbox_targets=bbox_targets.view(batch_size,height,width,A*4).permute(0,3,1,2).contiguous()
        outputs.append(bbox_targets)

        anchors_count=bbox_inside_weights.size(1)
        bbox_inside_weights=bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size,anchors_count,4)

        bbox_inside_weights=bbox_inside_weights.contiguous().view(batch_size,height,width,4*A).permute(0,3,1,2).contiguous()

        outputs.append(bbox_inside_weights)

        bbox_outside_weights=bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size,anchors_count,4)
        bbox_outside_weights=bbox_outside_weights.reshape(batch_size,height,width,4*A).permute(0,3,1,2).contiguous()
        outputs.append(bbox_outside_weights)

        return outputs

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass


def _unmap(data,count,inds,batch_size,fill=0):
    if data.dim()==2:
        ret=torch.Tensor(batch_size,count).fill_(fill).type_as(data)
        ret[:,inds]=data
    else:
        ret=torch.Tensor(batch_size,count,data.size(2)).fill_(fill).type_as(data)
        ret[:,inds,:]=data
    return ret

def _compute_targets_batch(ex_rois, gt_rois):
    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
