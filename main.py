from models.FASTER_RCNN import Faster_RCNN
from helper.reader import Reader
from helper.runner import Runner
import logging
import os
import random
import sys
import time
import argparse
import numpy as np
import torch.random

logger = logging.getLogger('logger')
def get_args():
    parser=argparse.ArgumentParser(description='arguments of program')
    parser.add_argument('--data_directory', type=str, default='./data', help='original data directory')
    parser.add_argument('--checkpoints_directory', type=str, default='./checkpoints', help='checkpoints directory')

    parser.add_argument('--best_checkpoint', type=bool, default=False, help='checkpoint to test')
    parser.add_argument('--dataset_name', type=str, default='tt100k_2021', help='the name of data set')
    parser.add_argument('--base_name',type=str,default='ResNet50',help='the basic bone of Faster-RCNN')
    # the path of data set should be {data_directory}/{dataset_name}
    # the model would be saved to {checkpoints_directory}/{base_name}_{dataset_name}/{current epoch}

    parser.add_argument('--stage', type=str, default='train', help='train/test')
    parser.add_argument('--logging_directory',
                        type=str, default='./log', help='the directory the log would be saved to')
    # the path of logging should be {logging_directory}/{base_name}_{dataset_name}_{daytime}
    parser.add_argument('--random_state', type=int, default=2021, help='the random seed')

    # arguments for runner
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='cuda:0 / cpu')
    parser.add_argument('--epoch', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--test_epoch', type=int, default=-1,
                        help='test with a period of some epoch (-1 means no test)')
    parser.add_argument('--stop', type=int, default=5, help='stop cnt when accuracy down continuously')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization in optimizer')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size while training/ validating')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='batch size while testing')

    # arguments for models
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='the drop out rate of linear layers')
    
    # rpn arguments
    parser.add_argument('--TRAIN.RPN_POSITIVE_WEIGHT', type=float, default=-1.0, help='')
    parser.add_argument('--USE_GPU_NMS', type=bool, default=True, help='')
    parser.add_argument('--TRAIN.RPN_PRE_NMS_TOP_N', type=int, default=12000, help='Number of top scoring boxes before using')
    parser.add_argument('--TEST.RPN_PRE_NMS_TOP_N', type=int, default=6000, help='Number of top scoring boxes before using')
    parser.add_argument('--TRAIN.RPN_POST_NMS_TOP_N', type=int, default=2000, help='Number of top scoring boxes after using')
    parser.add_argument('--TRAIN.RPN_POST_NMS_TOP_N', type=int, default=300, help='Number of top scoring boxes after using')
    parser.add_argument('--TRAIN.RPN_NMS_THRESH', type=float, default=0.7, help='NMS threshold')
    parser.add_argument('--TRAIN.RPN_NMS_THRESH', type=float, default=0.7, help='NMS threshold')
    parser.add_argument('--TRAIN.RPN_MIN_SIZE', type=int, default=8, help='min_size')
    parser.add_argument('--TRAIN.RPN_MIN_SIZE', type=int, default=16, help='min_size')
    parser.add_argument('--TRAIN.RPN_CLOBBER_POSITIVES', type=bool, default=False, help='')
    parser.add_argument('--TRAIN.RPN_NEGATIVE_OVERLAP', type=float, default=0.3, help='')
    parser.add_argument('--TRAIN.RPN_FG_FRACTION', type=float, default=0.5, help='')
    parser.add_argument('--TRAIN.RPN_POSITIVE_OVERLAP', type=float, default=0.7, help='')
    parser.add_argument('--TRAIN.RPN_BATCHSIZE', type=int, default=256, help='')

    args, unknown = parser.parse_known_args()

    # setting of paths
    setattr(args,'logging_file',os.path.join(
        args.logging_directory,
        '{}_{}_{}.txt'.format(args.base_name,
                              args.dataset_name,
                              time.strftime('%Y.%m.%d', time.localtime()))
    ))
    setattr(args,'dataset_path',os.path.join(
        args.data_directory,args.dataset_name
    ))
    setattr(args,'checkpoints_prefix',
            args.checkpoints_directory+'//'+'{}_{}'.format(args.base_name,args.dataset_name))

    return args

def main(args):
    """
    :return:
    """
    # set logger
    global logger
    logger.setLevel(level=logging.INFO)
    file_handler = logging.FileHandler(args.logging_file_name)
    file_handler.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.info('{}: start to logging\n'.format(time.strftime('%Y.%m.%d_%H:%M:%S', time.localtime())))

    logger.info('set random seed')
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    torch.cuda.manual_seed(args.random_state)
    torch.backends.cudnn.deterministic = True

    logger.info('build reader')
    reader=Reader(args)
    logger.info('build model')
    model=Faster_RCNN(reader.classes,args)
    logger.info('build runner')
    runner=Runner(args,reader,model)

    if args.stage=='train':
        logger.info('start to train')
        runner.train()
    else:
        logger.info('start to test')
        runner.evaluate('test',args.best_checkpoint)



if __name__=='__main__':
    args=get_args()
    main(args)
