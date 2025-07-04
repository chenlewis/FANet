from torch import nn
from torch.nn import functional as F
class Net(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, 2)

        # Conv1
        self.conv1 = nn.Conv2d(3,96,11,stride=4)

        # Conv2
        self.conv2 = nn.Conv2d(96,256,5,padding=2)

        # Conv3
        self.conv3 = nn.Conv2d(256,384,3, padding=1)

        # Conv4
        self.conv4 = nn.Conv2d(384,384, 3, padding=1)

        # Conv5
        self.conv5 = nn.Conv2d(384,256, 3, padding=1)

        # fc1
        self.fc1 = nn.Linear(1024,4096)

        # fc2
        self.fc2 = nn.Linear(4096, 128)

        #fc3
        self.fc3 = nn.Linear(128, self.num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1) # Flatten

        x = self.fc1(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc2(x)
        x_f = self.relu(x)
        x = F.dropout(x_f, p=0.5, training=self.training)

        x = self.fc3(x)
        # x = F.sigmoid(x)
        return x_f, x

if __name__ == '__main__':
    net = Net(2)