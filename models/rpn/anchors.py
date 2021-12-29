import numpy as np


def _whctrs(anchor):
    w = anchor[2]-anchor[0]+1
    h = anchor[3]-anchor[1]+1
    x_center = (anchor[0]+anchor[2])*0.5
    y_center = (anchor[1]+anchor[3])*0.5
    return w, h, x_center, y_center

# create anchors


def _make_anchors(ws, hs, x_center, y_center):
    # add one axis
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_center-0.5*(ws-1)),
                        (y_center-0.5*(hs-1)),
                        (x_center+0.5*(ws-1)),
                        (y_center+0.5*(ws-1)))
    return anchors


def _ratio_enum(anchor, ratios):
    w, h, x_center, y_center = _whctrs(anchor)
    size_list = w*h/ratios
    ws = np.round(np.sqrt(size_list))
    hs = np.round(ws*ratios)
    anchors = _make_anchors(ws, hs, x_center, y_center)
    return anchors


def _scale_enum(anchor, scales):
    w, h, x_center, y_center = _whctrs(anchor)
    ws = w*scales
    hs = h*scales
    anchors = _make_anchors(ws, hs, x_center, y_center)
    return anchors


def generate_anchors(base_size=16, ratios=np.array([0.5, 1, 2]), scales=2**np.arange(3, 6)):
    base_anchor = np.array([1, 1, base_size, base_size])-1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                        for i in range(ratio_anchors.shape[0])])
    return anchors
