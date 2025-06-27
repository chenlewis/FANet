import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from nets import Net
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torchvision.models import resnet18

os.environ['CUDA_VISIBLE_DEVICES']='0'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
learning_rate = 0.001

net = resnet18(pretrained=True)
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, 2)
net = net.to(device)

train_data_set = ImageFolder(r"D:\Face_IQA\dataset\CASIA-FASD\cbsr_antispoofing\frame\face\CNN\train", transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))]))
test_data_set = ImageFolder(r"D:\Face_IQA\dataset\CASIA-FASD\cbsr_antispoofing\frame\face\CNN\val", transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))]))
train_data_loader = data.DataLoader(train_data_set, batch_size = 128, shuffle = True, num_workers = 0, pin_memory = True)
test_data_loader = data.DataLoader(test_data_set, batch_size = 128, shuffle = False, num_workers = 0, pin_memory = True)

# optimizer = torch.optim.SGD(list(net.parameters()), lr=learning_rate)
optimizer = torch.optim.Adam(list(net.parameters()), lr=learning_rate)

def train(net, dataloader, optimizer, epoch):
    net.train()
    criterion = nn.CrossEntropyLoss()
    best_loss = 10000
    for e in range(epoch):
        print("epoch %d "%(e))
        for batch_idx, (X, y) in tqdm(enumerate(dataloader)):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = net(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        print("training loss: {:.4f}".format(loss.item()))
        '''
        val(net,test_data_loader, e)
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(net.state_dict(), "models/model_{}.pt".format(e))
        '''

def val(net,dataloader, e):
    net.eval()
    scores = []
    all_x_f = []
    all_y = []
    for batch_idx, (X, y) in tqdm(enumerate(dataloader)):
        X, y = X.to(device), y.to(device).view(-1, )
        optimizer.zero_grad()
        output = net(X)
        scores.append(F.sigmoid(output).detach().cpu().numpy()[:, 1:])
        # all_x_f.append(x_f.detach().cpu().numpy())
        all_y.append(y.cpu().numpy())

if __name__ == '__main__':
    train(net, train_data_loader, optimizer, 50)