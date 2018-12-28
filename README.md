# PointCapsNet

## 训练
* 无数据增强训练
`python run.py --epoch 50` <br>
* 若要在训练中加入旋转数据增强
`python run.py --epoch 50 --rotation xx,yy` <br>
xx和yy均为两位数，是点云旋转角度的范围，如`01,60`是将训练数据随机旋转1~60度<br>
* 若要接着上次的训练
`python run.py --epoch 50` --pretrain ./experiment/checkpoints/xx.pth <br>

## 测试
* 无旋转测试
`python evaluation.py <br>
* 若要旋转测试
`python evaluation.py --rotation xx,yy` <br>
xx和yy均为两位数，是点云旋转角度的范围，如`01,60`是将训练数据随机旋转1~60度

## 其他参数：
   ` 
   
    parser.add_argument('--mesh_size', default=24,
                        help='size when transfer pointcloud to mesh')       
                        
    parser.add_argument('--batchsize', default=32,
                        help='batch size in training')
                        
    parser.add_argument('--epoch',  default=5,
                        help='number of epoch in training')
                        
    parser.add_argument('--learning_rate', default=0.0001,
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
                        
    parser.add_argument('--data_path', type=str, default='./data/modelnet40_ply_hdf5_2048/',
                        help='data path')
                        
    parser.add_argument('--log_dir', type=str, default='./experiment/logs/',
                        help='decay rate of learning rate')
                        
    parser.add_argument('--num_class', type=str, default=40,
                        help='total number of class in classification')
                        
    parser.add_argument('--decay_rate', type=str, default=0.9,
                        help='decay rate of learning rate')
                        
    parser.add_argument('--rotation',  default=None,
                        help='range of training rotation')`
