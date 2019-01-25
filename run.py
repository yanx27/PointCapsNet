import matplotlib.pyplot as plt
import torch
import os
from torch.autograd import Variable
from tqdm import tqdm
from collections import defaultdict
from PointCapsNet import PointCapsNet
from optimizer import PointCapsNetLoss,exponential_decay
from DataLoader import load_data, myDataset
from utils import test, save_checkpoint, plot_loss_curve, plot_acc_curve
import argparse
from pathlib import Path
import logging
import datetime


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('CapsNet')
    parser.add_argument('--mesh_size', default=24,type=int,
                        help='size of mesh when transfer from pointcloud')
    parser.add_argument('--batchsize', type=int, default=32,
                        help='batch size in training')
    parser.add_argument('--epoch',  default=5, type=int,
                        help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0001, type=float,
                        help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    parser.add_argument('--train_metric', type=str, default=False,
                        help='whether evaluate on training dataset')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='whether use pretrain model')
    parser.add_argument('--result_dir', type=str, default='./experiment/results/',
                        help='dir to save pictures')
    parser.add_argument('--n_routing_iter', type=str, default=3,
                        help='Number of routing')
    parser.add_argument('--data_path', type=str, default='./data/shapenet16/',
                        help='data path')
    parser.add_argument('--log_dir', type=str, default='./experiment/logs/',
                        help='decay rate of learning rate')
    parser.add_argument('--num_class', type=int, default=16,
                        help='total number of class in classification')
    parser.add_argument('--decay_rate', type=float, default=0.9,
                        help='decay rate of learning rate')
    parser.add_argument('--rotation',  default=None,
                        help='range of training rotation')
    return parser.parse_args()

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    MESH_SIZE = args.mesh_size
    BATCHSIZE = args.batchsize
    LEARNING_RATE = args.learning_rate
    INPUT_SIZE = (1, MESH_SIZE, MESH_SIZE, MESH_SIZE)
    EPOCH = args.epoch
    COMPUTE_TRAIN_METRICS = args.train_metric
    ROUNTING_ITER = args.n_routing_iter
    DATA_PATH = args.data_path
    NUM_CLASS = args.num_class
    DECAY_RATE = args.decay_rate
    if args.rotation is not None:
        ROTATION = (int(args.rotation[0:2]),int(args.rotation[3:5]))
    else:
        ROTATION = None

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
    file_handler = logging.FileHandler(args.log_dir + 'train-'+ str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M'))+'.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    train_data, train_label, test_data, test_label = load_data(DATA_PATH)
    logger.info("The number of training data is: %d",train_data.shape[0])
    logger.info("The number of test data is: %d", test_data.shape[0])
    trainDataset = myDataset(train_data, train_label,meshmode = "density",rotation=ROTATION)
    if ROTATION is not None:
        print('The range of training rotation is',ROTATION)
    testDataset = myDataset(test_data, test_label, meshmode = "density",rotation=ROTATION)
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCHSIZE, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=BATCHSIZE, shuffle=False)

    '''MODEL LOADING'''
    model = PointCapsNet(INPUT_SIZE, NUM_CLASS, ROUNTING_ITER).cuda()
    if args.pretrain is not None:
        print('Use pretrain model...')
        logger.info('Use pretrain model')
        checkpoint = torch.load(args.pretrain)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('No existing model, starting training from scratch...')
        start_epoch = 0

    print(model)
    print('Number of Parameters: %d' % model.n_parameters())

    criterion = PointCapsNetLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-08
    )
    global_epoch = 0
    global_step = 0
    best_tst_accuracy = 0.0
    history = defaultdict(lambda: list())

    '''TRANING'''
    logger.info('Start training...')
    total_train_acc = []
    total_test_acc = []
    for epoch in range(start_epoch, EPOCH):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, EPOCH))
        logger.info('Epoch %d (%d/%s):' ,global_epoch + 1, epoch + 1, EPOCH)

        for batch_id, (x, y) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer = exponential_decay(optimizer, LEARNING_RATE, global_epoch, 1, DECAY_RATE)

            x = Variable(x).float().cuda()
            y = Variable(y.squeeze()).cuda()
            y_pred, x_reconstruction = model(x, y)

            loss, margin_loss, reconstruction_loss = criterion(x, y, x_reconstruction, y_pred.cuda(), NUM_CLASS)

            history['margin_loss'].append(margin_loss.cpu().data.numpy())
            history['reconstruction_loss'].append(reconstruction_loss.cpu().data.numpy())
            history['loss'].append(loss.cpu().data.numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        train_metrics, train_hist_acc = test(model, trainDataLoader) if COMPUTE_TRAIN_METRICS else (None, [])
        test_metrics, test_hist_acc = test(model, testDataLoader)
        total_train_acc += train_hist_acc
        total_test_acc += test_hist_acc

        print('Margin Loss: %f' % history['margin_loss'][-1])
        logger.info('Margin Loss: %f' , history['margin_loss'][-1])
        print('Reconstruction Loss: %f' % history['reconstruction_loss'][-1])
        logger.info('Reconstruction Loss: %f' , history['reconstruction_loss'][-1])
        print('Loss: %f' % history['loss'][-1])
        logger.info('Loss: %f' , history['loss'][-1])
        if COMPUTE_TRAIN_METRICS:
            print('Train Accuracy: %f' % (train_metrics['accuracy']))
            logger.info('Train Accuracy: %f' , (train_metrics['accuracy']))
        print('Test Accuracy: %f' % test_metrics['accuracy'])
        logger.info('Test Accuracy: %f' , test_metrics['accuracy'])

        # TODO show reconstruction mesh
        # idx = np.random.randint(0, len(x))
        # show_example(x[idx], y[idx], x_reconstruction[idx], y_pred[idx], args.result_dir, 'Epoch_{}'.format(epoch))

        if (test_metrics['accuracy'] >= best_tst_accuracy) and epoch > 5 == 0:
            best_tst_accuracy = test_metrics['accuracy']
            logger.info('Save model...')
            save_checkpoint(
                global_epoch + 1,
                train_metrics['accuracy'] if COMPUTE_TRAIN_METRICS else 0.0,
                test_metrics['accuracy'],
                model,
                optimizer,
                str(checkpoints_dir)
            )
        global_epoch += 1

    logger.info('End of training...')
    n_points_avg = 10
    n_points_plot = 1000
    plt.figure(figsize=(20, 10))
    plot_loss_curve(history, n_points_avg, n_points_plot, str(result_dir))
    plot_acc_curve(total_train_acc, total_test_acc, str(result_dir))


if __name__ == '__main__':
    args = parse_args()
    main(args)

