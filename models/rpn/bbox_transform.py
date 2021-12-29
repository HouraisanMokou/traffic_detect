import torch
import numpy as np


def bbox_transform_inv(boxes, deltas, batch_size):
    widths = boxes[:, :, 2]-boxes[:, :, 0]+1.0
    heights = boxes[:, :, 3]-boxes[:, :, 1]+1.0
    x_center = boxes[:, :, 0]+0.5*widths
    y_center = boxes[:, :, 1]+0.5*heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    # add one dimension on the third dimension
    pred_x_center = dx*widths.unsqueeze(2)+x_center.unsqueeze(2)
    pred_y_center = dy*heights.unsqueeze(2)+y_center.unsqueeze(2)
    pred_w = torch.exp(dw)*widths.unsqueeze(2)
    pred_h = torch.exp(dh)*heights.unsqueeze(2)

    pred_boxes = deltas.clone()

    # x1
    pred_boxes[:, :, 0::4] = pred_x_center-0.5*pred_w
    # x2
    pred_boxes[:, :, 1::4] = pred_y_center-0.5*pred_h
    # x3
    pred_boxes[:, :, 2::4] = pred_x_center+0.5*pred_w
    # x4
    pred_boxes[:, :, 3::4] = pred_y_center+0.5*pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape, batch_size):
    for i in range(batch_size):
        boxes[i, :, 0::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i, :, 1::4].clamp_(0, im_shape[i, 0]-1)
        boxes[i, :, 2::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i, :, 3::4].clamp_(0, im_shape[i, 0]-1)

    return boxes


def bbox_overlaps_batch(anchors, gt_boxes):
    batch_size = gt_boxes.shape[0]

    if anchors.dim() == 2:

        N = anchors.shape[0]
        K = gt_boxes.shape[1]

        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:, :, :4].contiguous()

        gt_boxes_x = (gt_boxes[:, :, 2]-gt_boxes[:, :, 0]+1)
        gt_boxes_y = (gt_boxes[:, :, 3]-gt_boxes[:, :, 1]+1)
        gt_boxes_area = (gt_boxes_x*gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:, :, 2]-anchors[:, :, 0]+1)
        anchors_boxes_y = (anchors[:, :, 3]-anchors[:, :, 1]+1)
        anchors_area = (anchors_boxes_x*anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(
            batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
              torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0])+1)

        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
              torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1])+1)
        ih[ih < 0] = 0

        ua = anchors_area+gt_boxes_area-(iw*ih)
        overlaps = iw*ih/ua

        # mask the overlap
        overlaps.masked_fill_(gt_area_zero.view(
            batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(
            batch_size, N, 1).expand(batch_size, N, K), -1)

    elif anchors.dim() == 3:
        N = anchors.shape[1]
        K = gt_boxes.shape[1]

        if anchors.shape[2] == 4:
            anchors = anchors[:, :, :4].contiguous()
        else:
            anchors = anchors[:, :, 1:5].contiguous()

        gt_boxes = gt_boxes[:, :, :4].contiguous()

        gt_boxes_x = (gt_boxes[:, :, 2]-gt_boxes[:, :, 0]+1)
        gt_boxes_y = (gt_boxes[:, :, 3]-gt_boxes[:, :, 1]+1)
        gt_boxes_area = (gt_boxes_x*gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:, :, 2]-anchors[:, :, 0]+1)
        anchors_boxes_y = (anchors[:, :, 3]-anchors[:, :, 1]+1)
        anchors_area = (anchors_boxes_x*anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(
            batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
              torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0])+1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
              torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1])+1)
        ih[ih < 0] = 0
        ua = anchors_area+gt_boxes_area-(iw*ih)

        overlaps = iw*ih/ua

        overlaps.masked_fill_(gt_area_zero.view(batch_size,1,K).expand(batch_size,N,K),0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size,N,1).expand(batch_size,N,K),-1)

    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps

def bbox_transform_batch(ex_rois,gt_rois):

    if ex_rois.dim()==2:
        ex_widths=ex_rois[:,2]-ex_rois[:,0]+1.0
        ex_heights=ex_rois[:,3]-ex_rois[:,1]+1.0
        ex_x_center=ex_rois[:,0]+0.5*ex_widths
        ex_y_center=ex_rois[:,1]+0.5*ex_heights

        gt_widths=gt_rois[:,:,2]-gt_rois[:,:,0]+1.0
        gt_heights=gt_rois[:,:,3]-gt_rois[:,:,1]+1.0
        gt_x_center=gt_rois[:,:,0]+0.5*gt_widths
        gt_y_center=gt_rois[:,:,1]+0.5*gt_heights

        targets_dx=(gt_x_center-ex_x_center.view(1,-1).expand_as(gt_x_center))/ex_widths
        targets_dy=(gt_y_center-ex_y_center.view(1,-1).expand_as(gt_y_center))/ex_heights
        targets_dw=torch.log(gt_widths/ex_widths.view(1,-1).expand_as(gt_widths))
        targets_dh=torch.log(gt_heights/ex_heights.view(1,-1).expand_as(gt_heights))
    elif ex_rois.dim()==3:
        ex_widths=ex_rois[:,:,2]-ex_rois[:,:,0]+1.0
        ex_heights=ex_rois[:,:,3]-ex_rois[:,:,1]+1.0
        ex_x_center=ex_rois[:,:,0]+0.5*ex_widths
        ex_y_center=ex_rois[:,:,1]+0.5*ex_heights

        gt_widths=gt_rois[:,:,2]-gt_rois[:,:,0]+1.0
        gt_heights=gt_rois[:,:,3]-gt_rois[:,:,1]+1.0
        gt_x_center=gt_rois[:,:,0]+0.5*gt_widths
        gt_y_center=gt_rois[:,:,1]+0.5*gt_heights

        targets_dx=(gt_x_center-ex_x_center)/ex_widths
        targets_dy=(gt_y_center-ex_y_center)/ex_heights
        targets_dw=torch.log(gt_widths/ex_widths)
        targets_dh=torch.log(gt_heights/ex_heights)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets=torch.stack((targets_dx,targets_dy,targets_dw,targets_dh),2)

    return targets