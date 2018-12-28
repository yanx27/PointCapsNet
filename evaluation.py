from PointCapsNet import PointCapsNet
from DataLoader import load_data, myDataset
from utils import test
import argparse
from pathlib import Path
import logging
import datetime
import os
import torch
import numpy as np

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('CapsNet')
    parser.add_argument('--mesh_size', default=24,
                        help='size when transfer pointcloud to mesh')
    parser.add_argument('--batchsize', default=32,
                        help='batch size in training')
    parser.add_argument('--rotation',  default=None,
                        help='range of test rotation')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    parser.add_argument('--result_dir', type=str, default='./experiment/results/',
                        help='dir to save pictures')
    parser.add_argument('--data_path', type=str, default='./data/modelnet40_ply_hdf5_2048/',
                        help='data path')
    parser.add_argument('--log_dir', type=str, default='./experiment/logs/',
                        help='decay rate of learning rate')
    parser.add_argument('--num_class', type=str, default=40,
                        help='total number of class in classification')
    parser.add_argument('--n_routing_iter', type=str, default=3,
                        help='Number of routing')

    return parser.parse_args()

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    MESH_SIZE = int(args.mesh_size)
    BATCHSIZE = int(args.batchsize)
    INPUT_SIZE = (1, MESH_SIZE, MESH_SIZE, MESH_SIZE)
    ROUNTING_ITER = args.n_routing_iter
    DATA_PATH = args.data_path
    NUM_CLASS = int(args.num_class)
    ROTATION = (int(args.rotation[0:2]),int(args.rotation[3:5]))


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
    logger = logging.getLogger("PointCapsNet")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.log_dir+ 'test-'+str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M'))+'.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------EVALUATION---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    _, _, test_data, test_label = load_data(DATA_PATH)
    logger.info("The number of test data is: %d", test_data.shape[0])
    testDataset = myDataset(test_data, test_label, meshmode = "density", rotation=ROTATION)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=BATCHSIZE, shuffle=False)

    '''MODEL LOADING'''
    model = PointCapsNet(INPUT_SIZE, NUM_CLASS, ROUNTING_ITER).cuda()
    print('Use pretrain model...')
    logger.info('Use pretrain model')
    checkpoints_list = os.listdir(str(checkpoints_dir))
    score = []
    for i in range(len(checkpoints_list)):
        score.append(float(checkpoints_list[i][11:17]))
    score = np.array(score)
    best_checkpoints = checkpoints_list[np.argmax(score)]
    try :
        checkpoint = torch.load(str(checkpoints_dir)+'/'+best_checkpoints)
        print('The original test accuracy is: %f'% checkpoint['test_accuracy'])
        logger.info('The original test accuracy is:',checkpoint['test_accuracy'])
    except:
        OSError('cannot find checkpoints file!!')

    model.load_state_dict(checkpoint['model_state_dict'])

    print(model)
    print('Number of Parameters: %d' % model.n_parameters())

    test_metrics, test_hist_acc = test(model, testDataLoader)
    print('Experimental test Accuracy: %f' % test_metrics['accuracy'])
    logger.info('Experimental test Accuracy: %f', test_metrics['accuracy'])



if __name__ == '__main__':
    args = parse_args()
    main(args)
