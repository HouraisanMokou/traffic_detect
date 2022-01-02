import copy
import os.path
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torch.utils.data.dataloader import DataLoader
from typing import NoReturn, List
import json
import imagesize
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

class PrefetchLoader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Reader():
    """
    to read the data

    feed_dict for dataloader:
        use dict to load (convenient for later use)
        should contain:
            im_data,im_info,gt_boxes,num_box
            gt_boxes: ground true roi and bbox (0-3bbox, 4target)
            im_info: w,h,scale
    """

    def __init__(self, args):
        self.dataset = args.dataset_name
        self.dataset_path = args.dataset_path
        self.w = 800
        self.h = 600

        self.preprocess()

    def preprocess(self) -> NoReturn:
        """
        preprocess the file (dataset path)
        :return:
        """
        if self.dataset == 'tt100k_2021':
            file_path = os.path.join(self.dataset_path, 'annotations_all.json')
            with open(file_path, 'rb') as f:
                data = json.load(f)

            self.classes = data['types']
            self._assign_ids()
            imgs = data['imgs']

            # for img_id in imgs:
            self.max_num=0
            for img_id in tqdm(imgs, desc='find max len',
                               leave=False, ncols=100, mininterval=0.01):
                objects = imgs[img_id]['objects']
                self.max_num=max(self.max_num,len(objects))
            # self.max_num = min(self.max_num, 1)
            train_l = []
            test_val_l = []
            li = ['new_train', 'new_test', 'new_other']  #
            flag=False
            for l in li:
                p = os.path.join('../data/tt100k_2021/', l)
                if not os.path.exists(p):
                    flag=True
                    os.makedirs(p)
            # for img_id in imgs:
            for img_id in tqdm(imgs, desc='preprocess target',
                               leave=False, ncols=100, mininterval=0.01):
                path = imgs[img_id]['path']
                new_path = os.path.join('../data/tt100k_2021/', path)
                objects = imgs[img_id]['objects']
                if not flag:
                    # w, h = imagesize.get(new_path)
                    w, h = imagesize.get(os.path.join('../data/tt100k_2021/', 'new_' + path))
                else:
                    img = Image.open(new_path)
                    w, h = img.size
                    img = img.resize((self.w, self.h))
                num_box = len(objects)
                gt_boxes = []
                for o in objects:
                    bbox = o['bbox']
                    category = o['category']
                    gt_box = [
                        bbox['xmin'] / w * self.w,
                        bbox['xmax'] / w * self.w,
                        bbox['ymin'] / h * self.h,
                        bbox['ymax'] / h * self.h,
                        self.class_dict[category]
                    ]
                    gt_boxes.append(gt_box)
                gt_boxes = np.array(gt_boxes)
                # gt_boxes = np.array(gt_boxes)[:self.max_num,:]
                # num_box=min(self.max_num,num_box)
                gt_boxes=np.pad(gt_boxes,((0,self.max_num-num_box),(0,0)))
                piece = {
                    'path': os.path.join('../data/tt100k_2021/', 'new_' + path),
                    'im_info': torch.from_numpy(np.array([w, h])),
                    'gt_boxes': torch.from_numpy(gt_boxes),
                    'num_box': torch.from_numpy(np.array([num_box]))
                }
                if flag:
                    img.save(os.path.join('../data/tt100k_2021/', 'new_' + path))
                if 'train' in path:
                    train_l.append(piece)
                elif 'test' in path:
                    test_val_l.append(piece)
            tes_val_len = len(test_val_l)
            split = int(np.ceil(tes_val_len * 0.8))
            test_l = test_val_l[:split]
            val_l = test_val_l[split:]
            self.data = {
                'train': train_l,
                'test': test_l,
                'val': val_l
            }

    def _assign_ids(self):
        ids = 1
        self.class_dict = {'bg': 0}
        for c in self.classes:
            self.class_dict[c] = ids
            ids += 1

    def get_loader(self, phase: str, batch_size: int) -> DataLoader:
        """
        :param phase: train or test (may be has val set
        :param batch_size: the batch size of dataloader
        :return: a dataloader for train or test
        """
        data = self.data[phase]
        image_set=ImageSet(data)
        return DataLoader(image_set, batch_size=256, num_workers=16, collate_fn=ImageSet._collate)

class ImageSet(VisionDataset):
    def __init__(self, data):
        super(ImageSet, self).__init__(root='')
        self.datas = data
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # target_transform=

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx: int):
        data = self.datas[idx]
        img = Image.open(data['path'])
        d = copy.deepcopy(data)
        d['im_data'] = self.transform(img)
        return d

    @staticmethod
    def _collate(dicts:List[dict]):
        # t1 = time.time()
        # data = [(d['im_data'], d['im_info'], d['gt_boxes'],d['num_box']) for d in dicts]
        # data = list(zip(*data))
        r= {
            'im_data':torch.stack([d['im_data'] for  d in dicts]),
            'im_info': torch.stack([d['im_info'] for  d in dicts]),
            'gt_bbox': torch.stack([d['gt_boxes'] for  d in dicts]),
            'num_box': torch.stack([d['num_box'] for  d in dicts])
        }
        # print(time.time()-t1)
        return r


if __name__ == '__main__':
    class args:
        dataset_name = 'tt100k_2021'
        dataset_path = '../data/tt100k_2021'


    r = Reader(args)
    md = ImageSet(r.data['train'])
    dl = PrefetchLoader(md,batch_size=256, num_workers=8, collate_fn=ImageSet._collate)#
    #
    import time

    t1 = time.time()
    t0=t1
    cnt = 0
    for d in dl:
        for k in d:
            d[k].to('cuda:1')
        cnt += 1
        t2=time.time()
        print( t2- t1, cnt)
        sys.stdout.flush()
        t1=t2
        # if cnt == 15:
        #     break
    print(time.time() - t0,cnt)