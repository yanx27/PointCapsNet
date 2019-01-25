import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from DataLoader import PartDataset
from PointCapsNetSeg import PointNetSeg
import torch.nn.functional as F
import datetime
import logging
from DataLoader import load_segdata
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser('PointCapsNetSeg')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--data_path', type=str, default='./data/shapenet16/', help='data path')
    parser.add_argument('--result_dir', type=str, default='./experiment/results/',help='dir to save pictures')
    parser.add_argument('--log_dir', type=str, default='./experiment/logs/',help='decay rate of learning rate')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')

    return parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(exist_ok=True)
    checkpoints_dir = Path('./experiment/checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("PointCapsNetSeg")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.log_dir + 'train-'+ str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M'))+'.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)
    DATA_PATH = args.data_path
    train_data, train_label, test_data, test_label = load_segdata(DATA_PATH)
    logger.info("The number of training data is: %d",train_data.shape[0])
    logger.info("The number of test data is: %d", test_data.shape[0])

    dataset = PartDataset(train_data,train_label)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                             shuffle=True, num_workers=int(args.workers))

    test_dataset = PartDataset(test_data,test_label)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize,
                                                 shuffle=True, num_workers=int(args.workers))

    num_classes = 50

    classifier = PointNetSeg(k=num_classes)
    blue = lambda x: '\033[94m' + x + '\033[0m'

    if args.pretrain is not None:
        classifier.load_state_dict(torch.load(args.pretrain))

    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    classifier.cuda()

    num_batch = len(dataset) / args.batchSize

    for epoch in range(args.epoch):
        for i, data in enumerate(dataloader, 0):
            points, target = data
            points, target = Variable(points), Variable(target.long())
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0]
            loss = F.nll_loss(pred, target)
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            # print('[%d: %d/%d] train loss: %f accuracy: %f' % (
            # epoch, i, num_batch, loss.item(), correct.item() / float(args.batchSize * 2500)))

            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points, target = Variable(points), Variable(target.long())
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred, _ = classifier(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0]
                # reduction.py line 154
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
                epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(args.batchSize * 2500)))

        torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (checkpoints_dir, epoch))


if __name__ == '__main__':
    args = parse_args()
    main(args)

