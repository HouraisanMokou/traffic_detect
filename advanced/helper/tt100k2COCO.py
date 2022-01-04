import os
import json
import random

import numpy as np
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


def trans(file, out):
    with open(os.path.join(file, 'annotations_all.json'), 'r') as f:
        data = json.load(f)
        classes = data['types']
        imgs = data['imgs']

    train_set = dict()
    test_set = dict()
    val_set = dict()

    train_set['info'] = info
    test_set['info'] = info
    val_set['info'] = info
    train_set['licenses'] = licenses
    test_set['licenses'] = licenses
    val_set['licenses'] = licenses

    ids = 0
    class_dict = dict()
    # class_dict = {'bg': 0}
    for c in classes:
        class_dict[c] = ids
        ids += 1

    obj_id = 1

    l=len(imgs)
    trl=int(np.floor(l*0.75))
    tl = int(np.floor(l * 0.2))

    idxs=np.random.permutation(l)
    trids=idxs[:trl]
    tids=idxs[trl:trl+tl]

    train_set = {'info': {}, 'licenses': [], 'categories': [], 'images': [], 'annotations': []}
    test_set = {'info': {}, 'licenses': [], 'categories': [], 'images': [], 'annotations': []}
    val_set = {'info': {}, 'licenses': [], 'categories': [], 'images': [], 'annotations': []}
    cnt=0
    for img_id in tqdm(imgs,desc='converting',total=len(imgs)):
        img_e = imgs[img_id]
        p = img_e['path']

        im = Image.open(os.path.join(file, p))
        w, h = im.size

        # if 'train' in p:
        #     dataset=train_set
        # elif 'test' in p:
        #     dataset=test_set
        # else:
        #     dataset=val_set

        im = im.resize((new_w, new_h))
        new_p = 'new/' + p.split('/')[-1]
        im.save(os.path.join(file, new_p))

        if cnt in trids:
            whole_set=train_set
        elif cnt in tids:
            whole_set=test_set
        else:
            whole_set=val_set

        whole_set['images'].append({
            'file_name': new_p,
            'id': img_id,
            'width': new_w,
            'height': new_h
        })
        objects = img_e['objects']

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
                    'area':width*height,
                    'bbox':[x,y,width,height],
                    'category_id':class_dict[o['category']],
                    'id':obj_id,
                    'image_id':img_id,
                    'iscrowd':0,
                    'segmentation':[[x,y,x+width,y,x+width,y+height,x,y+height]]
                }
            )
            obj_id+=1
        cnt+=1

    #
    for s in ['train','test','val']:
        with open(os.path.join(out,f'annotation_{s}.json'),'w') as f:
            json.dump(s,f,ensure_ascii=False,indent=1)

if __name__=='__main__':
    trans('../../data/tt100k_2021/','../../data/tt100k_2021/')
    from torchvision.datasets import CocoDetection
    from torch.utils.data.dataloader import DataLoader
    ds=CocoDetection(root='../../data/tt100k_2021/',annFile='../../data/tt100k_2021/annotation_train.json')
    dl=DataLoader(ds)
    for d in dl:
        print(d)
        break