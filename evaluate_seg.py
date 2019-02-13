#-*-coding:GBK -*-
import argparse
import os
import torch
import numpy as np
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
from torch.autograd import Variable
from DataLoader import PartDataset
from PointCapsNetSeg import PointNetSeg
import torch.nn.functional as F
import datetime
import logging
from DataLoader import load_segdata
from pathlib import Path
from utils import test_seg
from tqdm import tqdm
import importlib
import shutil
import sys

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def parse_args():
    parser = argparse.ArgumentParser('PointCapsNetSeg')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--data_path', type=str, default='./data/shapenet16/', help='data path')
    parser.add_argument('--experiment_path', type=str, default='', help='experiment path')
    parser.add_argument('--cnn_structure', type=str, default='UNet',
                        help='fill [CapsNet] or [UNet] when use_vox is True')
    parser.add_argument('--use_vox', type=bool, default=False, help='Whether use capsnet extract voxel feature or not')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--rotation',  default=None,help='range of training rotation')
    parser.add_argument('--n_routing_iter', type=int, default=1, help='Number if rounting iteration')

    return parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    '''CREATE DIR'''
    experiment_dir = './experiment/'+args.experiment_path
    print('Please check experiment direction is %s'%experiment_dir)
    log_dir = experiment_dir + '/logs'
    filelist = os.listdir(experiment_dir)

    for file in filelist:
        if file.find('.py') is not -1:
            model_name = file[:-3]

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/test_%s.txt'%model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------EVALUATION---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)
    DATA_PATH = args.data_path
    train_data, train_label, test_data, test_label = load_segdata(DATA_PATH)
    logger.info("The number of test data is: %d", test_data.shape[0])
    ROTATION = (int(args.rotation[0:2]), int(args.rotation[3:5])) if args.rotation is not None else None
    train_dataset = PartDataset(train_data,train_label,ROTATION)
    traindataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize,
                                                 shuffle=True, num_workers=int(args.workers))
    test_dataset = PartDataset(test_data,test_label,ROTATION)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize,
                                                 shuffle=True, num_workers=int(args.workers))
    num_classes = 50
    blue = lambda x: '\033[94m' + x + '\033[0m'

    USE_VOXEL_FEATURE = args.use_vox
    shutil.copy('%s/%s.py' % (experiment_dir,model_name), str(os.getcwd())+'/OldModel.py')
    model_module = importlib.import_module('OldModel')
    model = model_module.PointNetSeg(k=num_classes,
                                     n_routing=args.n_routing_iter,
                                     use_vox_feature=USE_VOXEL_FEATURE,
                                     cnn_structure = args.cnn_structure)
    model.cuda()
    checkpoint_dir = experiment_dir + '/checkpoints'
    model_list = os.listdir(checkpoint_dir)
    score = []
    for i in range(len(model_list)):
        score.append(float(model_list[i][-10:-4]))
    score = np.array(score)
    best_checkpoints = model_list[np.argmax(score)]
    pretrain = checkpoint_dir + '/' + best_checkpoints
    print('load model %s' % pretrain)
    model.load_state_dict(torch.load(pretrain))

    logger.info('load model %s' % pretrain)

    train_metrics, train_hist_acc, cat_mean_iou = test_seg(model, traindataloader, seg_label_to_cat)

    print('%s accuracy: %f  meanIOU: %f' % (blue('train'), train_metrics['accuracy'], train_metrics['iou']))
    logger.info('%s accuracy: %f  meanIOU: %f' % ('train', train_metrics['accuracy'], train_metrics['iou']))
    logger.info(cat_mean_iou)
    print(cat_mean_iou)
    
    test_metrics, test_hist_acc, cat_mean_iou = test_seg(model, testdataloader, seg_label_to_cat)

    print('%s accuracy: %f  meanIOU: %f' % (blue('test'), test_metrics['accuracy'], test_metrics['iou']))
    logger.info('%s accuracy: %f  meanIOU: %f' % ('test', test_metrics['accuracy'], test_metrics['iou']))
    logger.info(cat_mean_iou)
    print(cat_mean_iou)


if __name__ == '__main__':
    args = parse_args()
    main(args)
