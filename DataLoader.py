# *_*coding:utf-8 *_*
import numpy as np
import os
import h5py
import warnings
from pyntcloud import PyntCloud
import pandas as pd
import torch
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def load_data(dir):
    data_train0, label_train0 = load_h5(dir + 'ply_data_train0.h5')
    data_train1, label_train1 = load_h5(dir + 'ply_data_train1.h5')
    data_train2, label_train2 = load_h5(dir + 'ply_data_train2.h5')
    data_train3, label_train3 = load_h5(dir + 'ply_data_train3.h5')
    data_train4, label_train4 = load_h5(dir + 'ply_data_train4.h5')
    data_test0, label_test0 = load_h5(dir + 'ply_data_test0.h5')
    data_test1, label_test1 = load_h5(dir + 'ply_data_test1.h5')
    train_data = np.concatenate([data_train0,data_train1,data_train2,data_train3,data_train4])
    train_label = np.concatenate([label_train0,label_train1,label_train2,label_train3,label_train4])
    test_data = np.concatenate([data_test0,data_test1])
    test_label = np.concatenate([label_test0,label_test1])
    print('The shape of training data is:',train_data.shape)
    print('The shape of test data is:',test_data.shape)
    return train_data, train_label,test_data,test_label


class myDataset(Dataset):
    def __init__(self, pts, labels, aug = True, meshmode = "binary",mesh_size = 24, rotation = None):
        self.pts = pts
        self.labels = labels
        self.aug = aug
        self.meshmode = meshmode
        self.rotation = rotation
        self.mesh_size = mesh_size

    def __len__(self):
        return len(self.pts)

    def pointcloud2mesh(self, pointcloud, mode="binary", n=24):
        '''
        Generate mesh by point clouds
        :param pointcloud: Nx3 array original point clouds
        :param mode: method to generate mesh
        :param n: size of mesh
        :return: N*N*N array
        '''
        point = pd.DataFrame(pointcloud)
        point.columns = ['x', 'y', 'z']
        cloud = PyntCloud(point)
        voxelgrid_id = cloud.add_structure("voxelgrid", n_x=n, n_y=n, n_z=n)
        voxelgrid = cloud.structures[voxelgrid_id]
        binary_feature_vector = voxelgrid.get_feature_vector(mode = mode)
        return binary_feature_vector

    def rotate_point_cloud_by_angle(self, data, rotation_angle):
        """
        Rotate the point cloud along up direction with certain angle.
        :param batch_data: Nx3 array, original batch of point clouds
        :param rotation_angle: range of rotation
        :return:  Nx3 array, rotated batch of point clouds
        """
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_data = np.dot(data, rotation_matrix)

        return rotated_data

    def __getitem__(self, index):
        pointcloud = self.pts[index]
        if self.rotation is not None:
            angle = np.random.randint(self.rotation[0],self.rotation[1]) * np.pi / 180
            pointcloud = self.rotate_point_cloud_by_angle(pointcloud,angle)
        x = self.pointcloud2mesh(pointcloud,self.meshmode,n=self.mesh_size)
        x = x[np.newaxis,:,:,:]

        return x, self.labels[index]

if __name__ == '__main__':
    data_path = './data/modelnet40_ply_hdf5_2048/'
    h5_filename = os.path.join(data_path, 'ply_data_train0.h5')
    data, label = load_h5(h5_filename)
    print(len(np.unique(label)))
    # h5_filename = os.path.join(DATA_PATH, 'ply_data_train0.h5')
    # data, label = load_h5(h5_filename)
    # index10 = label <= NUM_CLASS-1
    # data = data[list(index10[:, 0]), :, :]
    # label = label[index10].reshape(-1, 1)
    # train_x = data[:int(len(data)*0.6), :, :]
    # train_y = label[:int(len(data)*0.6), ]
    # test_x = data[int(len(data)*0.6):, :, :]
    # test_y = label[int(len(data)*0.6):, ]
    # print('TRAINING DATA SHAPE:', train_x.shape)
    # print('TEST DATA SHAPE:', test_x.shape)
    myDataset = myDataset(data,label)
    myDataLoader = torch.utils.data.DataLoader(myDataset, batch_size=10, shuffle=False)
    for idx, (data, label) in enumerate(myDataLoader):
        print(data.shape)
        print(label.shape)
        if idx == 10:
            break
