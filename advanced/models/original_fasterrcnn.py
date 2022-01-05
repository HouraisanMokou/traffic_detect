import os.path

import numpy as np
import torch
from torch.optim import Adamax
from torchvision.models import mobilenet_v2
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN,resnet_fpn_backbone

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from tqdm import tqdm


def get_default(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def run_default(reader,model,logger,out):
    dl = reader.get_loader('train', 2)
    # model1 = mobilenet_v2(pretrained=True).features
    # model1.out_channels = 1280
    # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                    aspect_ratios=((0.5, 1.0, 2.0),))
    # roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
    #                                 output_size=7,
    #                                 sampling_ratio=2)
    # model = FasterRCNN(model1,
    #                    num_classes=len(class_dict),
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    o = Adamax([p for p in model.parameters() if p.requires_grad], lr=5e6)
    device='cuda:3'
    model.to(device)
    cnt=0
    eval_l = []
    dl2 = reader.get_loader('test', 2)
    for e in range(10):
        loss_l = []
        model.training()
        for d in tqdm(dl, desc='train', total=len(dl)):
            cnt += 1
            i, t = d
            for idx in range(len(i)):
                i[idx] = i[idx].to(device)

            for tt in t:
                for k in tt:
                    if isinstance(tt[k], torch.Tensor):
                        tt[k] = tt[k].to(device)
            output = model(i, t)

            losses = sum(loss for loss in output.values())
            loss_l.append(losses.cpu().detach().numpy())
            o.zero_grad()
            losses.backward()
            o.step()

            # if cnt%100==0:
            #     print(np.mean(loss_l))
        logger.info('train: epoch {}: {}'.format(e,np.mean(loss_l)))

        torch.save(model,os.path.join(out,f'epoch{e}.pth'))
        e_l = []
        model.evaluate()
        for d2 in dl2:
            evals=model(d2)
        eval_l.append()
    logger.info('f')