import os
import json
import random
import sys

import imagesize
import numpy as np
import torch
from PIL import Image

from tqdm import tqdm

info = {
    'year': 2021,
    'version': 'None',
    'description': 'None',
    'contributor': 'None',
    'url': 'None',
    'data_created': 'None',
}

licenses = {
    'id': 1,
    'name': 'None',
    'url': 'None'
}
new_w, new_h = 256, 256

class_dict = dict()


def trans(file, out):
    with open(os.path.join(file, 'annotations_all.json'), 'r') as f:
        data = json.load(f)
        classes = data['types']
        imgs = data['imgs']

    train_set = {'info': {}, 'licenses': [], 'categories': [], 'images': [], 'annotations': []}
    test_set = {'info': {}, 'licenses': [], 'categories': [], 'images': [], 'annotations': []}
    val_set = {'info': {}, 'licenses': [], 'categories': [], 'images': [], 'annotations': []}

    train_set['info'] = info
    test_set['info'] = info
    val_set['info'] = info
    train_set['licenses'] = licenses
    test_set['licenses'] = licenses
    val_set['licenses'] = licenses

    ids = 0

    # class_dict = {'bg': 0}
    for c in classes:
        class_dict[c] = ids
        ids += 1

    obj_id = 1

    l = len(imgs)
    trl = int(np.floor(l * 0.75))
    tl = int(np.floor(l * 0.2))

    idxs = np.random.permutation(l)
    trids = idxs[:trl]
    tids = idxs[trl:trl + tl]

    complete_set = {'info': {}, 'licenses': [], 'categories': [], 'images': [], 'annotations': []}
    cnt = 0
    flag = True if os.path.exists(os.path.join(file, 'new')) else False
    if flag==False:
        os.makedirs(os.path.join(file, 'new'))
    for img_id in tqdm(imgs, desc='converting', total=len(imgs)):
        img_e = imgs[img_id]
        p = img_e['path']

        new_p = 'new/' + p.split('/')[-1]
        if not flag:
            im = Image.open(os.path.join(file, p))
            w, h = im.size
            im = im.resize((new_w, new_h))
            im.save(os.path.join(file, new_p))
        else:
            w, h = imagesize.get(os.path.join(file, p))

        if cnt in trids:
            whole_set = train_set
        elif cnt in tids:
            whole_set = test_set
        else:
            whole_set = val_set

        objects = img_e['objects']
        if len(objects)>0:
            whole_set['images'].append({
                'file_name': new_p,
                'id': int(img_id),
                'width': new_w,
                'height': new_h
            })
            complete_set['images'].append({
                'file_name': new_p,
                'id': int(img_id),
                'width': new_w,
                'height': new_h
            })


            for o in objects:
                xmin = o['bbox']['xmin'] / w * new_w
                xmax = o['bbox']['xmax'] / w * new_w
                ymin = o['bbox']['ymin'] / h * new_h
                ymax = o['bbox']['ymax'] / h * new_h

                x = xmin
                y = ymin

                width = xmax - xmin
                height = ymax - ymin

                whole_set['annotations'].append(
                    {
                        'area': width * height,
                        'bbox': [x, y, width, height],
                        'category_id': class_dict[o['category']],
                        'labels': class_dict[o['category']],
                        'id': obj_id,
                        'image_id': int(img_id),
                        'iscrowd': 0,
                        'segmentation': [[x, y, x + width, y, x + width, y + height, x, y + height]],
                        'boxes': [x, y, xmax, ymax]
                    }
                )
                complete_set['annotations'].append(
                    {
                        'area': width * height,
                        'bbox': [x, y, width, height],
                        'category_id': class_dict[o['category']],
                        'labels': class_dict[o['category']],
                        'id': obj_id,
                        'image_id': int(img_id),
                        'iscrowd': 0,
                        'segmentation': [[x, y, x + width, y, x + width, y + height, x, y + height]],
                        'boxes': [x, y, xmax, ymax]
                    }
                )
                obj_id += 1
            cnt += 1

    #
    for s in ['train', 'test', 'val', 'complete']:
        with open(os.path.join(out, f'annotation_{s}.json'), 'w') as f:
            json.dump(eval(s + '_set'), f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    import torchvision
    print(torch.__version__,torchvision.__version__ )
    trans('../../data/tt100k_2021/', '../../data/tt100k_2021/')
    from torchvision.datasets import CocoDetection
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign
    from torchvision.models import mobilenet_v2
    from torchvision.transforms import ToTensor
    from torch.utils.data.dataloader import DataLoader
    from torchvision.transforms import functional as F
    from torch.optim import Adamax
    class tr():
        def __call__(self, pic,targets):
            img=F.to_tensor(pic)
            for target in targets:
                target['boxes']=torch.Tensor(target['boxes'])
            #print(pic, targets)
            return img,targets

        def __repr__(self):
            return self.__class__.__name__ + '()'
    ds = CocoDetection(root='../../data/tt100k_2021/'
                       , annFile='../../data/tt100k_2021/annotation_complete.json'
                       , transforms=tr())
    print(ds)
    dl = DataLoader(ds, batch_size=1)
    model1= mobilenet_v2(pretrained=True).features
    model1.out_channels=1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    model = FasterRCNN(model1,
                       num_classes=len(class_dict),
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    o=Adamax([p for p in model.parameters() if p.requires_grad],lr=1e5)
    model.to('cuda:2')
    for d in tqdm(dl,desc='train',total=len(dl)):
        i,t=d
        i=i.to('cuda:2')
        for obj in t:
            for k in obj:
                if isinstance(obj[k],torch.Tensor):
                    obj[k]=obj[k].to('cuda:2')
        output=model(i,t)

        losses = sum(loss for loss in output.values())
        o.zero_grad()
        losses.backward()
        o.step()
    print()
    print('f')