import pickle
import time

import numpy as np
import torch

from helper.reader import Reader
from torch import optim

from tqdm import tqdm


class Runner():
    """
    run, methods of training and testing
    """

    def __init__(self, args, reader: Reader, model):
        from main import logger
        self.reader = reader
        self.epoches = args.epoch
        self.test_epoch = args.test_epoch
        self.stop = args.stop
        self.lr = args.lr
        self.l2 = args.l2
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.device = args.device
        self.checkpoints_prefix = args.checkpoints_prefix
        self.model = model

        self.optimizer = None

    def evaluate(self, phase, file_name):
        self.model.load_state_dict(torch.load(file_name))
        ap, t = self.predict(phase)
        logger.info('test on model[{}]: ap[{}], time[{}]'.format(file_name, ap, t))

    def train(self):
        logger.info('start to train')
        self.model.to(self.device)
        loss_l, time_l, eval_list = [], [], []
        for epoch in self.epoches:
            batch_loss, used_time = self.fit(epoch)
            torch.save(self.model.state_dict(), self.checkpoints_prefix + f'_{epoch}.pth')
            loss_l.append(batch_loss)
            time_l.append(used_time)
            logger.info('epoch {}: loss[{}], used time[{}]'.format(epoch, batch_loss, used_time))
            if self.test_epoch != -1 and epoch % self.test_epoch == 0:
                ap, t = self.predict('val')
                logger.info('[test on val in epoch {}: ap[{}], time[{}]]'.format(epoch, ap, t))
                eval_list.append((ap, t))
                # terminal if test ap is continuously go down
                if self.terminal(eval_list):
                    break

        with open(self.checkpoints_prefix + '_results.pkl') as f:
            pickle.dump({
                'loss_list': loss_l,
                'time_list': time_l
            }, f)
        logger.info('model saved')

    def fit(self, epoch):
        if self.optimizer == None:
            p = [{'params': list(), 'weight_decay': self.l2}, {'params': list(), 'weight_decay': 0}]
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if 'bias' in name:
                        p[1]['params'].append(param)
                    else:
                        p[0]['params'].append(param)
            self.optimizer = optim.Adamax(p, lr=self.lr)

        dataloader = self.reader.get_loader('train', self.batch_size)

        self.model.train()
        loss_l, t1 = [], time.time()
        # for feed_dict in dataloader:
        for batch in tqdm(dataloader, desc='epoch {}'.format(epoch), leave=False, ncols=100, mininterval=0.1):
            for k in batch:
                batch[k].to(self.device)
            self.model(batch)
            loss = self.model.loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_tmp = loss.detach().numpy() if self.device == 'cpu' else loss.cpu().detach().numpy()
            loss_l.append(loss_tmp)

        return np.mean(loss_l), time.time() - t1

    def predict(self, phase) -> (float, float):
        dataloader = self.reader.get_loader(phase, self.eval_batch_size)

        batch_cnt = 0
        total_ap = 0
        t1 = time.time()
        result_list = []
        has_gt = False
        with torch.no_grad():
            # for batch in dataloader
            for batch in tqdm(dataloader, desc='testing', leave=False, ncols=100, mininterval=0.1):
                for k in batch:
                    batch[k].to(self.device)
                cls_prob, bbox_pred, rois_data = self.model(batch)
                result_list.append((cls_prob, bbox_pred, rois_data))
                if batch['gt_boxes'] is not None:
                    has_gt = True
        if has_gt:
            for cls_prob, bbox_pred, rois_data in result_list:
                total_ap += self.get_ap(cls_prob, bbox_pred, batch['gt_boxes'])
                batch_cnt += 1
        return total_ap / batch_cnt if has_gt else -1, time.time() - t1

    def terminal(self, eval_list):
        last = -np.inf
        for i in range(len(eval_list)):
            if i == self.stop:
                return True
            pos = len(eval_list) - 1 - i
            ap = eval_list[pos][0]
            if 2e-4 < last - ap:
                return False
            last = ap
        return False

    def get_ap(self, cls_pred, bbox_pred, gt) -> float:
        """
        calculate iou and than return ap
        threshold=0.5
        :return:
        """
