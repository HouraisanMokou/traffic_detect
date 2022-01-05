import copy
import os.path
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset, CocoDetection
from torch.utils.data.dataloader import DataLoader
from typing import NoReturn, List
import json
import imagesize
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from helper.tt100k2COCO import trans,class_dict


class PrefetchLoader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())



class Reader():
    def __init__(self, args):
        self.dataset = args.dataset_name
        self.dataset_path = args.dataset_path
        self.w = 256
        self.h = 256

        self.com = transforms.Compose([
            transforms.ToTensor()
        ])

        self.preprocess()
        self.classes=class_dict

    def preprocess(self) -> NoReturn:
        """
        preprocess the file (dataset path)
        :return:
        """
        if self.dataset == 'tt100k_2021':
            trans(self.dataset_path, self.dataset_path)

    def get_loader(self, phase: str, batch_size: int) -> DataLoader:
        """
        :param phase: train or test (may be has val set
        :param batch_size: the batch size of dataloader
        :return: a dataloader for train or test
        """
        md = CocoDetection(root=self.dataset_path,
                           annFile=os.path.join(self.dataset_path, f'annotation_{phase}.json'),
                           transform=self.com,
                           target_transform=Coco2target())
        return DataLoader(md, num_workers=2, batch_size=batch_size,collate_fn=collate)


class Coco2target:
    def __call__(self, target):
        objs = target  # ['annotation']
        boxes = torch.as_tensor([obj['bbox'] for obj in objs]).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]

        labels = torch.as_tensor([obj['category_id'] for obj in objs], dtype=torch.int64)
        target = {
            'boxes': boxes,
            'labels': labels
        }
        return target


def collate(batch):
    i,t=list(zip(*batch))
    return list(i),list(t)
    # print(batch[0])
    # i, t = tuple(zip(*batch))
    # return i[0], t[0]


if __name__ == '__main__':
    import torchvision

    print(torch.__version__, torchvision.__version__)
    from torchvision.datasets import CocoDetection
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign
    from torchvision.models import mobilenet_v2
    from torchvision.transforms import ToTensor
    from torch.utils.data.dataloader import DataLoader
    from torchvision.transforms import functional as F
    from torch.optim import Adamax


    class args:
        dataset_name = 'tt100k_2021'
        dataset_path = '../../data/tt100k_2021'


    r = Reader(args)

    dl = r.get_loader('train', 2)
    model1 = mobilenet_v2(pretrained=True).features
    model1.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                    output_size=7,
                                    sampling_ratio=2)
    model = FasterRCNN(model1,
                       num_classes=len(class_dict),
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    o = Adamax([p for p in model.parameters() if p.requires_grad], lr=1e5)
    device='cuda:3'
    model.to(device)
    cnt=0
    for e in range(10):
        loss_l = []
        for d in dl:#tqdm(dl, desc='train', total=len(dl)):
            cnt += 1
            i, t = d
            for idx in range(len(i)):
                i[idx] = i[idx].to(device)

            for tt in t:
                for k in tt:
                    if isinstance(tt[k], torch.Tensor):
                        tt[k] = tt[k].to(device)
            output = model(i, t)
            print(output)
            losses = sum(loss for loss in output.values())
            loss_l.append(losses.cpu().detach().numpy())
            o.zero_grad()
            losses.backward()
            o.step()

            # if cnt%100==0:
            #     print(np.mean(loss_l))
        print('epoch {}: {}'.format(e,np.mean(loss_l)))
    print('f')
    # class args:
    #     dataset_name = 'tt100k_2021'
    #     dataset_path = '../../data/tt100k_2021'
    #
    #
    # r = Reader(args)
    # from torchvision.transforms import functional as F
    #
    # md = CocoDetection(root=args.dataset_path, annFile=os.path.join(args.dataset_path, 'annotation_val.json')
    #                    , transform=com, target_transform=Coco2target())
    # i, t = md.__getitem__(0)
    # dl = DataLoader(md, num_workers=2, batch_size=256, collate_fn=collate)
    # cnt = 0
    # for d in dl:
    #     i, t = d
    #     i = i.to('cuda:2')
    #     for k in t:
    #         if isinstance(t[k], torch.Tensor):
    #             t[k] = t[k].to('cuda:2')
    #     print(torch.cuda.memory_allocated(device='cuda:2') / (1024 ** 3))
    #     cnt += 1
    #     if cnt == 1:
    #         break
