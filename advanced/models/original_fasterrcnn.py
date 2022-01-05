import os.path

import numpy as np
import torch
from torch.optim import Adamax, SGD
from torchvision.models import mobilenet_v2
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone
import pickle

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from tqdm import tqdm


def get_default(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_default2(num_classes):
    model1 = mobilenet_v2(pretrained=True).features
    # model1 = resnet_fpn_backbone('resnet50',False)
    # model1.out_channels = 256
    anchor_generator = AnchorGenerator(sizes=((1, 2, 4, 8),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                    output_size=7,
                                    sampling_ratio=2)
    model = FasterRCNN(model1,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model


def run_default(reader, model, logger, out):
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

    o = SGD([p for p in model.parameters() if p.requires_grad], lr=0.001,
            momentum=0.2, weight_decay=0.0005)
    # o = SGD([p for p in model.parameters() if p.requires_grad], lr=1e6)
    device = 'cuda:3'
    model.to(device)
    cnt = 0
    eval_l = []
    losses_l = []
    dl2 = reader.get_loader('test', 2)
    dl3 = reader.get_loader('val', 2)
    for e in range(1):
        loss_l = []
        model.train()
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
            # print(output)
            losses = sum(loss for loss in output.values())
            loss_l.append(losses.cpu().detach().numpy())
            o.zero_grad()
            losses.backward()
            o.step()

            # if cnt%100==0:
            #     print(np.mean(loss_l))
        tmp_l = np.mean(loss_l)
        losses_l.append(tmp_l)
        logger.info('train: epoch {}: {}'.format(e, tmp_l))

        torch.save(model, os.path.join(out, f'epoch{e}.pth'))
        model.eval()
        e_l = []
        for d in dl2:#tqdm(dl2, desc='test', total=len(dl2)):
            i, t = d
            for idx in range(len(i)):
                i[idx] = i[idx].to(device)

            for tt in t:
                for k in tt:
                    if isinstance(tt[k], torch.Tensor):
                        tt[k] = tt[k].to(device)
            output = model(i, t)
            for ou in output:
                for k in ou:
                    ou[k] = ou[k].cpu().detach().numpy()
            e_l.append(output)
            print(output)

            # if cnt%100==0:
            #     print(np.mean(loss_l))
        eval_l.append(e_l)
    e_l2 = []
    for d in tqdm(dl3, desc='test', total=len(dl3)):
        i, t = d
        for idx in range(len(i)):
            i[idx] = i[idx].to(device)

        for tt in t:
            for k in tt:
                if isinstance(tt[k], torch.Tensor):
                    tt[k] = tt[k].to(device)
        output = model(i, t)
        for ou in output:
            for k in ou:
                ou[k] = ou[k].cpu().detach().numpy()
        e_l2.append(output)

        # if cnt%100==0:
        #     print(np.mean(loss_l))
    # e_l = []
    # model.evaluate()
    # for d2 in dl2:
    #     evals=model(d2)
    # eval_l.append()

    with open(os.path.join(out, 'result.pkl'), 'wb') as f:
        data = {
            'train': losses_l,
            'test': eval_l,
            'val': e_l2
        }
        pickle.dump(data, f)
    logger.info('f')
