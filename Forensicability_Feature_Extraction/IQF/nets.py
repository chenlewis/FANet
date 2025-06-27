from torch import nn
from torch.nn import functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        # fc1
        self.fc1 = nn.Linear(94, 64)
        # fc2
        self.fc2 = nn.Linear(64, 32)
        #fc3
        self.fc3 = nn.Linear(32, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc2(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3(x)
        # x = F.sigmoid(x)
        return x

if __name__ == '__main__':
    net = Net()