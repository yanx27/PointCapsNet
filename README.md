# PointCapsNet验证点云的旋转不变性

## 训练：
* 无数据增强训练
`python run.py --epoch 50` <br>
* 在训练中加入旋转数据增强
`python run.py --epoch 50 --rotation xx,yy` <br>
xx和yy均为两位数，是点云旋转角度的范围，如`01,60`是将训练数据随机旋转1~60度<br>
* 接着上次的训练
`python run.py --epoch 50 --pretrain ./experiment/checkpoints/xx.pth `<br>
* 训练数据放在`./data/modelnet40_ply_hdf5_2048`
* 实验模型保存在`./experiment/checkpoints/`
* 实验记录保存在`./experiment/logs/`
* 实验结果图保存在`./experiment/results/`

## 测试：
* 会自动选取`./experiment/checkpoints/`中正确率最高的模型进行测试
* 无旋转测试
`python evaluation.py` <br>
* 旋转测试
`python evaluation.py --rotation xx,yy` <br>
xx和yy均为两位数，是点云旋转角度的范围，如`01,60`是将训练数据随机旋转1~60度

## 网络结构：
* 运行`python PointCapsNet.py` 

## 语义分割任务：
* 运行`python train_seg.py` 
- 网络结构
    - [x] Pointnet做点云的特征提取
    - [ ] CapsNet做体素的特征提取
    
## 其他参数：
   ` 
   
    parser.add_argument('--mesh_size', default=24,
                        help='点云转换成体素时候的尺寸，默认24x24x24')       
                        
    parser.add_argument('--batchsize', default=32,
                        help='batch size in training')
                        
    parser.add_argument('--epoch',  default=5,
                        help='number of epoch in training')
                        
    parser.add_argument('--learning_rate', default=0.0001,
                        help='learning rate in training')
                        
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
                        
    parser.add_argument('--train_metric', type=str, default=False,
                        help='在训练的时候是否测试训练集上的正确率')
                        
    parser.add_argument('--pretrain', type=str, default=None,
                        help='是否接着预训练的模型接着训练')
    parser.add_argument('--result_dir', type=str, default='./experiment/results/',
                        help='dir to save pictures')
                        
    parser.add_argument('--n_routing_iter', type=str, default=3,
                        help='Number of routing')
                        
    parser.add_argument('--data_path', type=str, default='./data/modelnet40_ply_hdf5_2048/',
                        help='data path')
                        
    parser.add_argument('--log_dir', type=str, default='./experiment/logs/',
                        help='decay rate of learning rate')
                        
    parser.add_argument('--num_class', type=str, default=40,
                        help='分类的总数（防止类别过多无法训练）')
                        
    parser.add_argument('--decay_rate', type=str, default=0.9,
                        help='decay rate of learning rate')
                        
    parser.add_argument('--rotation',  default=None,
                        help='训练时旋转数据增强的角度范围')`
