# *_*coding:utf-8 *_*
import numpy as np
import os
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 3D Convolution layer
class Conv3d_1(torch.nn.Module):
    def __init__(self, in_channels, out_channels=256, kernel_size=9): #TODO change kenel_size in other value
        super(Conv3d_1, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels, 128, kernel_size)
        self.conv2 = torch.nn.Conv3d(128, out_channels, kernel_size)
        self.BN1 = torch.nn.BatchNorm3d(128)
        self.BN2 = torch.nn.BatchNorm3d(out_channels)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.activation(x)
        return x

# Primary Capsules
class PrimaryCapsules(torch.nn.Module):
    def __init__(self, input_shape=(256, 16, 16, 16), capsule_dim=8,
                 out_channels=32, kernel_size=9, stride=2):
        super(PrimaryCapsules, self).__init__()
        self.input_shape = input_shape
        self.capsule_dim = capsule_dim
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = self.input_shape[0]

        self.conv = torch.nn.Conv3d(
            self.in_channels,
            self.out_channels * self.capsule_dim,
            self.kernel_size,
            self.stride
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(-1, x.size()[1], x.size()[2], x.size()[3], self.out_channels, self.capsule_dim)
        return x

# Routing
class Routing(torch.nn.Module):
    def __init__(self, caps_dim_before=8, caps_dim_after=16,
                 n_capsules_before=(4 * 4 * 4 * 32), n_capsules_after=10):
        super(Routing, self).__init__()
        self.n_capsules_before = n_capsules_before
        self.n_capsules_after = n_capsules_after
        self.caps_dim_before = caps_dim_before
        self.caps_dim_after = caps_dim_after

        # Parameter initialization not specified in the paper
        n_in = self.n_capsules_before * self.caps_dim_before
        variance = 2 / (n_in)
        std = np.sqrt(variance)
        self.W = torch.nn.Parameter(
            torch.randn(
                self.n_capsules_before,
                self.n_capsules_after,
                self.caps_dim_after,
                self.caps_dim_before) * std,
            requires_grad=True)

    # Equation (1)
    @staticmethod
    def squash(s):
        s_norm = torch.norm(s, p=2, dim=-1, keepdim=True)
        s_norm2 = torch.pow(s_norm, 2)
        v = (s_norm2 / (1.0 + s_norm2)) * (s / s_norm)
        return v

    # Equation (2)
    def affine(self, x): # torch.Size([1, 16384, 8])
        x = self.W @ x.unsqueeze(2).expand(-1, -1, self.n_capsules_after, -1).unsqueeze(-1)
        return x.squeeze()

    # Equation (3)
    @staticmethod
    def softmax(x, dim=-1):
        exp = torch.exp(x)
        return exp / torch.sum(exp, dim, keepdim=True)

    def routing(self, u, r, l):
        b = Variable(torch.zeros(u.size()[0], l[0], l[1]), requires_grad=False).cuda()

        for iteration in range(r):
            c = Routing.softmax(b)
            s = (c.unsqueeze(-1).expand(-1, -1, -1, u.size()[-1]) * u).sum(1)
            v = Routing.squash(s)
            b += (u * v.unsqueeze(1).expand(-1, l[0], -1, -1)).sum(-1)
        return v

    def forward(self, x, n_routing_iter):
        x = x.view((-1, self.n_capsules_before, self.caps_dim_before))
        #print('step1', x.shape)
        x = self.affine(x)  # torch.Size([16384, 40, 16])
        #print('step2', x.shape)
        x = self.routing(x, n_routing_iter, (self.n_capsules_before, self.n_capsules_after))
        return x

class Norm(torch.nn.Module):
    def __init__(self):
        super(Norm, self).__init__()

    def forward(self, x):
        x = torch.norm(x, p=2, dim=-1)
        return x

# Decoder
class Decoder(torch.nn.Module):
    def __init__(self, in_features, out_features, output_size):
        super(Decoder, self).__init__()
        self.decoder = self.assemble_decoder(in_features, out_features)
        self.output_size = output_size

    def assemble_decoder(self, in_features, out_features):
        HIDDEN_LAYER_FEATURES = [512, 1024]
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, HIDDEN_LAYER_FEATURES[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_FEATURES[0], HIDDEN_LAYER_FEATURES[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_FEATURES[1], out_features),
            torch.nn.Sigmoid(),
        )

    def forward(self, x, y):
        x = x[np.arange(0, x.size()[0]), y.cpu().data.numpy(), :].cuda()
        x = self.decoder(x)
        x = x.view(*((-1,) + self.output_size))
        return x

# PointCapsNet
class PointCapsNet(torch.nn.Module):
    def __init__(self, input_shape, num_class, n_routing_iter=3,  use_reconstruction=True):
        super(PointCapsNet, self).__init__()
        assert len(input_shape) == 4

        self.input_shape = input_shape
        self.n_routing_iter = n_routing_iter
        self.use_reconstruction = use_reconstruction

        self.conv1 = Conv3d_1(input_shape[0], 256, 5)
        self.primary_capsules = PrimaryCapsules(
            input_shape=(256, 16, 16, 16),
            capsule_dim=8,
            out_channels=32,
            kernel_size=9,
            stride=2
        )
        self.routing = Routing(
            caps_dim_before=8,
            caps_dim_after=16,
            n_capsules_before=4 * 4 * 4 * 32,
            n_capsules_after=num_class
        )
        self.norm = Norm()

        if (self.use_reconstruction):
            self.decoder = Decoder(16, int(np.prod(input_shape)),input_shape)

    def n_parameters(self):
        return np.sum([np.prod(x.size()) for x in self.parameters()])

    def forward(self, x, y=None):
        conv1 = self.conv1(x)
        primary_capsules = self.primary_capsules(conv1)
        digit_caps = self.routing(primary_capsules, self.n_routing_iter)
        scores = self.norm(digit_caps)

        if (self.use_reconstruction and y is not None):
            reconstruction = self.decoder(digit_caps, y).view((-1,) + self.input_shape)
            return scores, reconstruction

        return scores

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((128,1,24,24,24)).cuda()
    print('input', input.shape)
    out = Conv3d_1(input.shape[1], 256, 5).cuda()(input)
    print('After cov1', out.shape)
    out = PrimaryCapsules(
            input_shape=(256, 16, 16, 16),
            capsule_dim=8,
            out_channels=32,
            kernel_size=9,
            stride=2).cuda()(out)
    print('After PrimaryCapsules', out.shape)
    out = Routing().cuda()(out,2)
    print('After Routing', out.shape)
    score = Norm()(out)
    print('After Norm', score.shape)
    decoder = Decoder(16, int(np.prod((1, 24, 24, 24))), (1, 24, 24, 24)).cuda()
    y = torch.IntTensor(np.array([np.random.randint(0, 10) for i in range(128)]))
    reconstruction = decoder(out, y).view((-1,) + (1, 24, 24, 24))
    print('After reconstruction', reconstruction.shape)
    # model = PointCapsNet((1,24,24,24), 3).cuda()
    # y_pred, x_reconstruction = model(input, y)
    # print('x shape',input.shape)
    # print('y shape',y.shape)
    # print(y_pred.shape)
    # print(x_reconstruction.shape)

