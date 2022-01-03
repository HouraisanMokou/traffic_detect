from models.roi.roi_pooling.functions.roi_pooling import roi_pooling_2d, roi_pooling_2d_pytorch
from torch import nn

class ROI(nn.Module):
    def __init__(self, output_h, output_w, spatial_scale):
        super(ROI, self).__init__()
        self.output_size = (output_h, output_w)
        self.spatial_scale = spatial_scale

    def forward(self, features, rois):
        assert (features.is_cuda() and rois.is_cuda()) # if cupy useless, use below pytorch version (much slower)
        return roi_pooling_2d(features, rois, self.output_size, spatial_scale=self.spatial_scale)
        # roi_pooling_2d_pytorch(features, rois, self.output_size, spatial_scale=self.spatial_scale)
