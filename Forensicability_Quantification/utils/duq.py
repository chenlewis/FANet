import torch
import torch.nn as nn
import numpy as np

from torchvision.models import resnet18

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

feature_extractor = resnet18()

class DUQ(nn.Module):
    def __init__(
        self,
        num_classes,
        batch_size,
        centroid_size,
        length_scale,
        eta,
    ):
        super().__init__()


        self.feature_extractor_q = Net(94, 1)
        self.feature_extractor_f = Net(128 ,1)
        self.eta = eta
        self.centroid_size = centroid_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.Lambda = nn.Parameter(
            torch.zeros(num_classes)
        )
        nn.init.kaiming_normal_(self.Lambda.unsqueeze(0), nonlinearity="relu")

        self.register_buffer("N", torch.zeros(num_classes) + batch_size//num_classes)
        e_c_init = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 0.5]])
        self.register_buffer(
            "m", torch.from_numpy(e_c_init)
        )
        self.m = self.m * self.N[:, None]

        self.sigma = length_scale

    def rbf(self, z_batch):
        e_c = self.m / self.N[:, None]
        diff = torch.zeros(self.batch_size, self.centroid_size, self.num_classes).cuda()
        for i, z in enumerate(z_batch):
            diff[i, :, 0] = self.Lambda[0] * (z - e_c[0])
            diff[i, :, 1] = self.Lambda[1] * (z - e_c[1])
            diff[i, :, 2] = self.Lambda[2] * (z - e_c[2])
        diff = (diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()

        return diff, e_c

    def update_ec(self, z, y):
        self.N = self.eta * self.N + (1 - self.eta) * y.sum(0)

        ec_sum = torch.zeros(self.num_classes, self.centroid_size).double().cuda()
        for i, z_i in enumerate(z):
            if (y[i].detach().cpu().numpy() == [1.0, 0.0, 0.0]).all():
                ec_sum[0] += z_i
            elif (y[i].detach().cpu().numpy() == [0.0, 1.0, 0.0]).all():
                ec_sum[1] += z_i
            elif (y[i].detach().cpu().numpy() == [0.0, 0.0, 1.0]).all():
                ec_sum[2] += z_i
        self.m = self.eta * self.m + (1 - self.eta) * ec_sum

    def forward(self, x_q, x_f):
        y_q = self.feature_extractor_q(x_q)
        y_f = self.feature_extractor_f(x_f)
        z = torch.cat((y_q, y_f), 1)
        y_pred, e_c = self.rbf(z)

        return y_q, y_f, z, y_pred, e_c, self.Lambda