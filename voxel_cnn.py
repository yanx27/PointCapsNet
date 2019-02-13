from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d
from torch.nn import ReLU, Sigmoid
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from torch.autograd import Variable
import torch

class Conv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
            Conv3d(inp_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU())

        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Deconv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):
        super(Deconv3D_Block, self).__init__()

        self.deconv = Sequential(
            ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel, kernel, kernel),
                            stride=(stride, stride, stride), padding=(padding, padding, padding), output_padding=0, bias=True),
            ReLU())

    def forward(self, x):
        return self.deconv(x)


class ChannelPool3d(AvgPool1d):

    def __init__(self, kernel_size, stride, padding):
        super(ChannelPool3d, self).__init__(kernel_size, stride, padding)
        self.pool_1d = AvgPool1d(self.kernel_size, self.stride, self.padding, self.ceil_mode)

    def forward(self, inp):
        n, c, d, w, h = inp.size()
        inp = inp.view(n, c, d * w * h).permute(0, 2, 1)
        pooled = self.pool_1d(inp)
        c = int(c / self.kernel_size[0])
        return inp.view(n, c, d, w, h)

class UNet3D(Module):
    def __init__(self, num_channels=32, output_channel = 32, feat_channels=[64, 128, 256, 512, 1024], residual='conv'):
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

        super(UNet3D, self).__init__()

        # Encoder downsamplers
        self.pool1 = MaxPool3d((2, 2, 2))
        self.pool2 = MaxPool3d((2, 2, 2))
        self.pool3 = MaxPool3d((2, 2, 2))
        self.pool4 = MaxPool3d((2, 2, 2))

        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv3d(feat_channels[0], output_channel, kernel_size=1, stride=1, padding=0, bias=True)

        # Activation function
        self.sigmoid = Sigmoid()

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
        vox2point_id = np.zeros((pcl_feature.size(0), pcl.size(1), pcl.size(2)))
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

        return Variable(torch.Tensor(pcl_feature)).cuda()

    def forward(self, x):
        # Encoder part
        # print('x', x.size())

        x1 = self.conv_blk1(x)
        # print('x1',x1.size())

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)
        # print('x2', x2.size())

        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)
        # print('x3', x3.size())

        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)
        # print('x4', x4.size())

        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)
        # print('base', base.size())

        # Decoder part

        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        # print('d_high3', x1.size())

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        # print('d_high2', d_high2.size())

        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)
        # print('d_high1', d_high1.size())

        seg = self.sigmoid(self.one_conv(d_high1))
        # print('seg', seg.size())

        return seg

if __name__ == '__main__':
    net = UNet3D(residual='pool').cuda()

    import torch

    x = torch.ones(1, 32, 32, 32, 32).cuda()
    out = net(x)
    print(out.size())

