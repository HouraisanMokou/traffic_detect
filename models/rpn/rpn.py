import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from config import cfg
from _ProposalLayer import _ProposalLayer
from anchor_target_layer import anchor_target_layer


class RPN(nn.Module):
    """
    rpn net
    """

    def __ini__(self, din, args):
        super(RPN, self).__init__()

        self.din = din  # the depth of input feature map
        self.anchor_scales = args.ANCHOR_SCALES
        self.anchor_ratios = args.ANCHOR_RATIOS
        self.feat_stride = args.FEAT_STRIDE[0]

        # in: self.din -> out: 512, kernel_size: (3,3)
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # 2k cls score layer
        # 2(backgroud/foreground)*9(anchors)
        self.nc_score_out = len(self.anchor_scales)*len(self.anchor_ratios)*2
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # 4k coordinate layer
        self.nc_bbox_out = len(self.anchor_scales) * \
            len(self.anchor_ratios)*4  # 4 coordinates
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        self.RPN_proposal=_ProposalLayer(self.feat_stride,self.anchor_scales,self.anchor_ratios)

        self.RPN_anchor_target=anchor_target_layer(self.feat_stride,self.anchor_scales,self.anchor_ratios)

        # # loss
        # self.cross_entropy = None
        # self.loss_box = None

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @property
    def loss(self):
        return self.cross_entropy+self.loss_box*10

    @property
    def reshape(x,d):
        input_shape=x.size()
        x=x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1]*input_shape[2])/float(d)),
            input_shape[3]
        )
        return x

    def forward(self,base_feat,im_info,gt_boxes,num_boxes):
        batch_size=base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1=F.relu(self.RPN_Conv(base_feat),inplace=True)

        # get rnp classification score
        rpn_cls_score=self.RPN_cls_score(rpn_conv1)

        new_score=self.reshape(rpn_cls_score,2)
        score_row_soft=F.softmax(new_score,1)
        rpn_cls_prob=self.reshape(score_row_soft,self.nc_score_out)

        # rpn offsets
        rpn_offsets=self.RPN_bbox_pred(rpn_conv1)

        if self.training:
            cfg_key='TRAIN'
        else:
            cfg_key='TEST'

        rois=self.RPN_proposal((rpn_cls_prob.data,rpn_offsets.data,im_info,cfg_key))

        self.rpn_loss_cls=0
        self.rpn_loss_box=0

        if self.training:
            # check whether gt_boxes is none
            assert gt_boxes is not None
            
            rpn_data=self.RPN_anchor_target((rpn_cls_score.data,gt_boxes,im_info,num_boxes))

            rpn_cls_score=score_row_soft.permute(0,2,3,1).contiguous().view(batch_size,-1,2)
            rpn_label=rpn_data[0].view(batch_size,-1)

            rpn_keep=Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score=torch.index_select(rpn_cls_score.view(-1,2),0,rpn_keep)
            rpn_label=torch.index_select(rpn_label.view(-1),0,rpn_keep.data)
            rpn_label=Variable(rpn_label.int())
            self.rpn_loss_cls=F.cross_entropy(rpn_cls_score,rpn_label)
            fg_cnt=torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights=rpn_data[1:]

            rpn_bbox_inside_weights=Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights=Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets=Variable(rpn_bbox_targets)

            self.rpn_loss_box=_smooth_l1_loss(rpn_offsets,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights,sigma=3,dim=[1,2,3])

        return rois,self.rpn_loss_cls,self.rpn_loss_box

def _smooth_l1_loss(bbox_pred,bbox_targets,bbox_inside_weights,bbox_outside_weights,sigma=1.0,dim=[1]):
    sigma_2=sigma**2
    box_diff=bbox_pred-bbox_targets
    in_box_diff=bbox_inside_weights*box_diff
    abs_in_box_diff=torch.abs(in_box_diff)
    smoothL1_sign=(abs_in_box_diff<1./sigma_2).detach().float()

    in_loss_box=torch.pow(in_box_diff,2)*(sigma_2/2.)*smoothL1_sign+(abs_in_box_diff-(0.5/sigma_2))*(1.-smoothL1_sign)
    out_loss_box=bbox_outside_weights*in_loss_box

    loss_box=out_loss_box
    for i in sorted(dim,reverse=True):
        loss_box=loss_box.sum(i)
    loss_box=loss_box.mean()
    return loss_box