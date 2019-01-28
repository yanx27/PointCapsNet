import argparse
import os
import torch
import torch.nn.parallel
import torch.optim as optim
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


seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def parse_args():
    parser = argparse.ArgumentParser('PointCapsNetSeg')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--data_path', type=str, default='./data/shapenet16/', help='data path')
    parser.add_argument('--result_dir', type=str, default='./experiment/results/',help='dir to save pictures')
    parser.add_argument('--log_dir', type=str, default='./experiment/logs/',help='decay rate of learning rate')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--train_metric', type=str, default=False, help='Whether evaluate on training data')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--rotation',  default=None,help='range of training rotation')

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
    ROTATION = (int(args.rotation[0:2]), int(args.rotation[3:5])) if args.rotation is not None else None

    dataset = PartDataset(train_data,train_label,rotation=ROTATION)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                             shuffle=True, num_workers=int(args.workers))

    test_dataset = PartDataset(test_data,test_label)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize,
                                                 shuffle=True, num_workers=int(args.workers))

    num_classes = 50
    blue = lambda x: '\033[94m' + x + '\033[0m'

    model = PointNetSeg(k=num_classes)
    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))

    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )
    model.cuda()
    history = defaultdict(lambda: list())
    best_acc = 0
    COMPUTE_TRAIN_METRICS = args.train_metric

    for epoch in range(args.epoch):
        for i, data in tqdm(enumerate(dataloader, 0),total=len(dataloader),smoothing=0.9):
            points, target = data
            points, target = Variable(points.float()), Variable(target.long())
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            model = model.train()
            pred, _ = model(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0]
            loss = F.nll_loss(pred, target)
            history['loss'].append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
        if COMPUTE_TRAIN_METRICS:
            train_metrics, train_hist_acc = test_seg(model, dataloader)
            print('Epoch %d  %s loss: %f accuracy: %f  meanIOU: %f' % (
                epoch, blue('train'), history['loss'][-1], train_metrics['accuracy'],train_metrics['iou']))
            logger.info('Epoch %d  %s loss: %f accuracy: %f  meanIOU: %f' % (
                epoch, 'train', history['loss'][-1], train_metrics['accuracy'],train_metrics['iou']))


        test_metrics, test_hist_acc = test_seg(model, testdataloader)

        print('Epoch %d  %s accuracy: %f  meanIOU: %f' % (
                 epoch, blue('test'), test_metrics['accuracy'],test_metrics['iou']))
        logger.info('Epoch %d  %s accuracy: %f  meanIOU: %f' % (
                 epoch, 'test', test_metrics['accuracy'],test_metrics['iou']))
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            torch.save(model.state_dict(), '%s/seg_model_%.3d_%.4f.pth' % (checkpoints_dir, epoch, best_acc))
            logger.info('Save model..')
            print('Save model..')


if __name__ == '__main__':
    args = parse_args()
    main(args)

