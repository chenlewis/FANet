import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from nets import Net
from tqdm import tqdm
from Data import MyDataset

os.environ['CUDA_VISIBLE_DEVICES']='0'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
learning_rate = 0.01

if use_cuda:
    net = Net().to(device)
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(net)

train_data_set = MyDataset("data/train.csv")
test_data_set = MyDataset("data/test.csv")
train_data_loader = data.DataLoader(train_data_set, batch_size = 128, shuffle = True, num_workers = 0, pin_memory = True)
test_data_loader = data.DataLoader(test_data_set, batch_size = 128, shuffle = False, num_workers = 0, pin_memory = True)

optimizer = torch.optim.Adam(list(net.parameters()), lr=learning_rate)

def train(net, dataloader, optimizer, epoch):
    net.train()
    for e in range(epoch):
        print("epoch %d "%(e))
        for batch_idx, (X, y) in tqdm(enumerate(dataloader)):
            X, y_q = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_q_pred = net(X)
            loss = F.mse_loss(y_q_pred, y_q.unsqueeze(1), reduction="mean")
            loss.backward()
            optimizer.step()

        print("training loss: {:.4f}".format(loss.item()))
        val(net,test_data_loader, e)
    torch.save(net.state_dict(), "models/model.pt")

def val(net,dataloader, e):
    net.eval()
    val_loss = 0
    all_y = []
    for batch_idx, (X, y) in tqdm(enumerate(dataloader)):
        X, y_q = X.to(device), y.to(device).view(-1, )
        optimizer.zero_grad()
        y_q_pred = net(X)
        all_y.append(F.sigmoid(y_q_pred).cpu().numpy())
        loss = F.mse_loss(y_q_pred, y_q.unsqueeze(1), reduction="mean")
        val_loss += loss.item()
    print("val loss: {:.4f}".format(val_loss / batch_idx))


if __name__ == '__main__':
    train(net, train_data_loader, optimizer, 50)