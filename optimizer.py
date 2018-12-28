import numpy as np
import torch
from torch.autograd import Variable

# Define Loss Functions
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

# Margin Loss
class MarginLoss(torch.nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, lamb=0.5):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lamb = lamb

    def forward(self, scores, y, num_class):
        y = Variable(to_categorical(y, num_class))

        Tc = y.float()
        loss_pos = torch.pow(torch.clamp(self.m_pos - scores, min=0), 2)
        loss_neg = torch.pow(torch.clamp(scores - self.m_neg, min=0), 2)
        loss = Tc * loss_pos + self.lamb * (1 - Tc) * loss_neg
        loss = loss.sum(-1)
        return loss.mean()

# Reconstruction Loss
class SumSquaredDifferencesLoss(torch.nn.Module):
    def __init__(self):
        super(SumSquaredDifferencesLoss, self).__init__()

    def forward(self, x_reconstruction, x):
        loss = torch.pow(x - x_reconstruction, 2).sum(-1).sum(-1)
        return loss.mean()

# Total Loss
class PointCapsNetLoss(torch.nn.Module):
    def __init__(self, reconstruction_loss_scale=0.0005):
        super(PointCapsNetLoss, self).__init__()
        self.digit_existance_criterion = MarginLoss()
        self.digit_reconstruction_criterion = SumSquaredDifferencesLoss()
        self.reconstruction_loss_scale = reconstruction_loss_scale

    def forward(self, x, y, x_reconstruction, y_pred, num_class):
        margin_loss = self.digit_existance_criterion(y_pred.cuda(), y, num_class)
        reconstruction_loss = self.reconstruction_loss_scale * \
                              self.digit_reconstruction_criterion(x_reconstruction, x)
        loss = margin_loss + reconstruction_loss
        return loss, margin_loss, reconstruction_loss

# Optimizer
def exponential_decay(optimizer, learning_rate, global_step, decay_steps, decay_rate, staircase=False):
    if (staircase):
        decayed_learning_rate = learning_rate * np.power(decay_rate, global_step // decay_steps)
    else:
        decayed_learning_rate = learning_rate * np.power(decay_rate, global_step / decay_steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = decayed_learning_rate

    return optimizer
