# PointCapsNet

## 环境
* [pyntcloud](https://github.com/daavoo/pyntcloud)
* pytorch 0.4.1

## 分类任务训练：
* 无数据增强训练
`python train_clf.py --epoch 50` <br>
* 在训练中加入旋转数据增强
`python train_clf.py --epoch 50 --rotation xx,yy` <br>
xx和yy均为两位数，是点云旋转角度的范围，如`01,60`是将训练数据随机旋转1~60度<br>
* 接着上次的训练
`python train_clf.py --epoch 50 --pretrain ./experiment/checkpoints/xx.pth `<br>
* 训练数据放在`./data/modelnet40_ply_hdf5_2048`
* 实验模型保存在`./experiment/checkpoints/`
* 实验记录保存在`./experiment/logs/`
* 实验结果图保存在`./experiment/results/`

## 分类任务测试：
* 会自动选取`./experiment/checkpoints/`中正确率最高的模型进行测试
* 无旋转测试
`python evaluate_clf.py` <br>
* 旋转测试
`python evaluate_clf.py --rotation xx,yy` <br>
xx和yy均为两位数，是点云旋转角度的范围，如`01,60`是将训练数据随机旋转1~60度

## 分类任务网络结构：
* 运行`python PointCapsNet.py` 
    
## 分类任务参数： 
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
                        
## 语义分割任务训练：
* 运行`python train_seg.py --epoch 50 --n_routing_iter 3 --use_vox True --rotation xx,yy` ，
   如果不输入--use_vox True则为PointNet
 * 模型的训练日志、.py文件和训练好的参数保存在当前的时间的文件夹，如`./experiment/2019-02-03_17-10/`
- 网络结构
    - [x] Pointnet做点云的特征提取
    - [x] CapsNet做体素的特征提取
    - [ ] 超参数调试
* 主要调试的超参数：

   1.  n_routing_iter： 在训练时CapsNet动态路由的迭代次数<br>
   2. 点云转体素前的特征维度 <br>
   3. 体素转点云后的特征维度 <br>
   4. 3D卷积和3D反卷积的卷积核尺寸、个数以及卷积层层数 <br>
   5. 学习率和学习率衰减等常规超参数
## 语义分割任务测试：
* 运行例如`python evaluate_seg.py --experiment_path 2019-02-03_17-10 --use_vox True`
* 会自动在`./experiment/2019-02-03_17-10/checkpoints/`里寻找最优的模型，测试结果保存在`./experiment/2019-02-03_17-10/logs/`
* 会返回 mean accuracy、mean IOU以及每个类别的IOU，如： <br>
![](pic.png)

## 语义分割任务参数：
    
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    
    parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
    
    parser.add_argument('--data_path', type=str, default='./data/shapenet16/', help='data path')
    
    parser.add_argument('--result_dir', type=str, default='./experiment/results/',help='dir to save pictures')
    
    parser.add_argument('--log_dir', type=str, default='./experiment/logs/',help='decay rate of learning rate')
    
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    
    parser.add_argument('--train_metric', type=bool, default=False, help='Whether evaluate on training data')
    
    parser.add_argument('--use_vox', type=bool, default=False, help='Whether use capsnet extract voxel feature or not')
    
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    
    parser.add_argument('--rotation',  default=None,help='range of training rotation')
    
    parser.add_argument('--n_routing_iter', type=int, default=1, help='Number if rounting iteration')`
