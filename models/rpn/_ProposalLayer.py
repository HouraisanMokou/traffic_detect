import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from models.rpn.anchors import generate_anchors
from models.rpn.bbox_transform import bbox_transform_inv,clip_boxes
# from config import cfg
from models.rpn.nms import nms

class _ProposalLayer(nn.Module):

    def __init__(self, feat_stride, scales, ratios,args) -> None:
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride

        # numpy -> Tensor
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales),
                                                        ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)
        self.args=args

        def forward(self, input):
        # rois=self.RPN_proposal((rpn_cls_prob.data,rpn_offsets.data,im_info,cfg_key))
        scores = input[0][:, self._num_anchors:, :, :]
        bbox_deltas = input[1]
        im_info = input[2]
        cfg_key = input[3]

        pre_nms_topN = eval(f'self.args.{cfg_key}_RPN_PRE_NMS_TOP_N')
        post_nums_topN = eval(f'self.args.{cfg_key}_RPN_POST_NMS_TOP_N')
        nms_thresh = eval(f'self.args.{cfg_key}_RPN_NMS_THRESH')
        min_size = eval(f'self.args.{cfg_key}_RPN_MIN_SIZE')

        batch_size = bbox_deltas.shape[0]

        feat_height, feat_width = scores.shape[2], scores.shape[3]
        shift_x = np.arange(0, feat_width)*self._feat_stride
        shift_y = np.arange(0, feat_height)*self._feat_stride

        # Return coordinate matrices from coordinate vectors.
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                            shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()

        A=self._num_anchors
        K=shifts.shape[0]

        self._anchors=self._anchors.type_as(scores)
        anchors=self._anchors.view(1,A,4)+shifts.view(K,1,4)
        anchors=anchors.view(1,K*A,4).expand(batch_size,K*A,4)

        bbox_deltas=bbox_deltas.permute(0,2,3,1).contiguous()
        bbox_deltas=bbox_deltas.view(batch_size,-1,4)

        scores=scores.permute(0,2,3,1).contiguous()
        scores=scores.view(batch_size,-1)

        proposals=bbox_transform_inv(anchors,bbox_deltas,batch_size)
        
        # clip predicted boxes to image
        proposals=clip_boxes(proposals,im_info,batch_size)

        scores_keep=scores
        proposals_keep=proposals
        _,order=torch.sort(scores_keep,1,True)

        output=scores.new(batch_size,pre_nms_topN,5).zero_()
        for i in range(batch_size):
            # using threshold to delete boxes
            proposals_single=proposals_keep[i]
            scores_single=scores_keep[i]

            order_single=order[i]
            
            # numel(): the number of elements
            if pre_nms_topN>0 and pre_nms_topN<scores_keep.numel():
                order_single=order_single[:pre_nms_topN]

            proposals_single=proposals_single[order_single,:]
            scores_single=scores_single[order_single].view(-1,1)
            
            keep_idx_i=nms(torch.cat((proposals_single,scores_single),1),nms_thresh,force_cpu=True)
            keep_idx_i=keep_idx_i.long().view(-1)

            if post_nums_topN>0:
                keep_idx_i=keep_idx_i[:post_nums_topN]
            
            proposals_single=proposals_single[keep_idx_i,:]
            scores_single=scores_single[keep_idx_i,:]

            num_proposal= proposals_single.size(0)
            output[i,:,0]=i
            output[i,:num_proposal,1:]=proposals_single
        return output
