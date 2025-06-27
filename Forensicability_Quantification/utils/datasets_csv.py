import torch
from torch.utils import data
import pandas as pd
import numpy as np

class MyDataset(data.Dataset):
    def __init__(self, csv_path):
        self.reader = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        x_q = self.reader.iloc[index, 1:95]
        x_f = self.reader.iloc[index, 95:223]
        y_q = self.reader.iloc[index, -3]
        y_f = self.reader.iloc[index, -2]
        labels = self.reader.iloc[index, -1]
        
        x_q = np.array([x_q])
        x_q = x_q.astype('float')
        x_q = torch.from_numpy(x_q)
        x_q = torch.as_tensor(x_q, dtype=torch.float32)
        x_q = x_q[0]

        x_f = np.array([x_f])
        x_f = x_f.astype('float')
        x_f = torch.from_numpy(x_f)
        x_f = torch.as_tensor(x_f, dtype=torch.float32)
        x_f = x_f[0]

        y_q = np.array([y_q])
        y_q = y_q.astype('float')
        y_q = torch.from_numpy(y_q)
        y_q = torch.as_tensor(y_q, dtype=torch.float32)
        y_q = y_q[0]

        y_f = np.array([y_f])
        y_f = y_f.astype('float')
        y_f = torch.from_numpy(y_f)
        y_f = torch.as_tensor(y_f, dtype=torch.float32)
        y_f = y_f[0]

        labels = np.array([labels])
        labels = labels.astype('int')
        labels = torch.from_numpy(labels)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        labels = labels[0]
        return x_q, x_f, y_q, y_f, labels

def main():
    batch_size = 128

    custom_data_from_csv = MyDataset('data/train.csv')
    train_loader = data.DataLoader(dataset=custom_data_from_csv, batch_size=batch_size, shuffle=False, drop_last=True)
            
    for inputData, (features, target) in enumerate(train_loader):
        print(inputData)
        print(features.shape)
        print(target.shape)
        print(target)

if __name__ == '__main__':
    main()