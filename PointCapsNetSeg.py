import numpy as np
import os
import pandas as pd
import torch
from torch.autograd import Variable
from pyntcloud import PyntCloud
import torch.nn as nn
import torch.nn.functional as F
from PointCapsNet import  Conv3d_1, PrimaryCapsules, Routing, Norm, Decoder
from DataLoader import load_h5

class T_Net(nn.Module):
    def __init__(self):
        super(T_Net, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat = True):
        super(PointNetEncoder, self).__init__()
        self.stn = T_Net()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans) # batch matrix multiply
        x = x.transpose(2,1)
        x = self.conv1(x)
        #print('x_skip',x_skip.shape)
        x = F.relu(self.bn1(x))
        pointfeat = x
        #print(pointfeat.size())
        x_skip = self.conv2(x)
        x = F.relu(self.bn2(x_skip))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, x_skip
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, x_skip

class CapsuleBlock(torch.nn.Module):
    '''
    Extract feature by capsnet
    :param input: BxNxC1, where C1 is number of input channel
    :return:  BxNxC2, where C2 is number of outpur channel
    '''
    def __init__(self, out_channels=256, kernel_size=5,num_class=50,n_routing_iter=1):
        super(CapsuleBlock, self).__init__()
        self.in_channels = 128
        self.n_routing_iter = n_routing_iter
        self.conv_encoder = Conv3d_1(self.in_channels, out_channels, kernel_size)
        self.primary_capsules = PrimaryCapsules(
            input_shape=(256, 16, 16, 16),
            capsule_dim=8,
            out_channels=32,
            kernel_size=9,
            stride=2
        )
        self.routing = Routing(
            caps_dim_before=8,
            caps_dim_after=64,
            n_capsules_before=4 * 4 * 4 * 32,
            n_capsules_after=num_class
        )
        self.norm = Norm()
        self.decoder1 = nn.ConvTranspose3d(50,128,3,3)
        self.decoder2 = nn.ConvTranspose3d(128,512,3,2,1,1)

    def pcl2vox(self, pcl, pcl_feature, n=24):
        '''
        Generate mesh feature by point clouds feature
        :param pcl: BxPx3 tensor original point clouds （P is the number of point in each pointcloud）
        :param pcl_feature: BxNxC tensor feature of point clouds
        :param N: size of mesh
        :return: BxNxNxNxC tensor
        '''
        self.in_channels = pcl_feature.shape[2]
        mesh_feature = np.zeros((pcl_feature.size(0), n, n, n, self.in_channels))
        vox2point_id = np.zeros((pcl_feature.size(0),pcl.size(1),pcl.size(2)))
        pcl_feature = pcl_feature.cpu().data.numpy()
        pcl = pcl.cpu().data.numpy()
        for i in range(len(mesh_feature)):
            point = pd.DataFrame(pcl[i], columns=['x', 'y', 'z'])
            cloud = PyntCloud(point)
            voxelgrid_id = cloud.add_structure("voxelgrid", n_x=n, n_y=n, n_z=n)
            voxelgrid = cloud.structures[voxelgrid_id]
            x_vox = voxelgrid.voxel_x
            y_vox = voxelgrid.voxel_y
            z_vox = voxelgrid.voxel_z
            vox2point_npid = np.concatenate([x_vox.reshape((-1, 1)), y_vox.reshape((-1, 1)), z_vox.reshape((-1, 1))],
                                            axis=1)
            vox2point_id[i] = vox2point_npid
            for j in range(vox2point_id.shape[1]):
                x, y, z = vox2point_npid[j]
                mesh_feature[i, x, y, z] += pcl_feature[i, j]

        return Variable(torch.Tensor(mesh_feature)).cuda(), vox2point_id.astype(int)

    def vox2pcl(self, vox2point_idx, vox_feature):
        '''
        :param vox2point_idx: the corresponding relationship between voxel and pointcloud
        :param vox_feature: BxNxNxNxC tensor
        :return: BxPx3 tensor （P is the number of point in each pointcloud）
        '''
        pcl_feature = np.zeros((vox_feature.size(0), vox2point_idx.shape[1], vox_feature.size(1)))
        vox_feature = vox_feature.cpu().data.numpy()
        for idx in range(len(vox2point_idx)):
            for i in range(vox2point_idx.shape[1]):
                x, y, z = vox2point_idx[idx, i]
                pcl_feature[idx, i, :] = vox_feature[idx, :, x, y, z]

        return Variable (torch.Tensor(pcl_feature)).cuda()

    def forward(self, pcl,pcl_feature,n=24):
        mesh_feature, vox2point_idx = self.pcl2vox(pcl,pcl_feature,n)
        mesh_feature  = mesh_feature.permute(0,4,1,2,3)
        conv1 = self.conv_encoder(mesh_feature)
        primary_capsules = self.primary_capsules(conv1)
        digit_caps = self.routing(primary_capsules, self.n_routing_iter)
        resize = digit_caps.view(digit_caps.size(0),digit_caps.size(1),4,4,4)
        decoder = nn.ReLU()(self.decoder1(resize))
        decoder = nn.ReLU()(self.decoder2(decoder))
        pcl2_feature = self.vox2pcl(vox2point_idx,decoder)
        return pcl2_feature

class PointNetSeg(nn.Module):
    def __init__(self, k = 2, n_routing=1, use_vox_feature = True):
        super(PointNetSeg, self).__init__()
        self.k = k
        self.use_vox_feature = use_vox_feature
        self.CapsuleBlock = CapsuleBlock(n_routing_iter = n_routing).cuda()
        self.feat = PointNetEncoder(global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088+512, 512, 1) if use_vox_feature else torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x, ):
        init_feature = x
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, x_skip = self.feat(x)
        if self.use_vox_feature:
            caps_feature = self.CapsuleBlock(init_feature.permute(0, 2, 1), x_skip.permute(0, 2, 1))
            x = torch.cat((x, caps_feature.permute(0, 2, 1)), 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    sim_data = Variable(torch.rand(32, 2048, 3))
    sim_data = sim_data.permute(0,2,1)
    # trans = T_Net()
    # out = trans(sim_data)
    # print('stn', out.size())
    #
    # pointfeat = PointNetEncoder(global_feat=True)
    # out, _ = pointfeat(sim_data)
    # print('global feat', out.size())
    #
    # pointfeat = PointNetEncoder(global_feat=False)
    # out, _ = pointfeat(sim_data)
    # print('point feat', out.size())

    SegNet = PointNetSeg(k=50)
    out, _ =  SegNet(sim_data)
    print('SegNet', out.size())

    data_path = './data/shapenet16/'
    h5_filename = os.path.join(data_path, 'ply_data_train0.h5')
    data, label, seg = load_h5(h5_filename)

    pcl_feature = Variable(torch.rand(32, 2048, 64)).cuda()
    pcl = Variable(torch.Tensor(data[0:36])).cuda()

    caps = CapsuleBlock().cuda()
    print('pcl',pcl.shape)
    print('pcl_feature',pcl_feature.shape)
    out = caps(pcl, pcl_feature, 24)

