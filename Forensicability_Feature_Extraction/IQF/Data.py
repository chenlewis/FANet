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
        y_q = self.reader.iloc[index, -1]
        
        x_q = np.array([x_q])
        x_q = x_q.astype('float')
        x_q = torch.from_numpy(x_q)
        x_q = torch.as_tensor(x_q, dtype=torch.float32)
        x_q = x_q[0]

        y_q = np.array([y_q])
        y_q = y_q.astype('float')
        y_q = torch.from_numpy(y_q)
        y_q = torch.as_tensor(y_q, dtype=torch.float32)
        y_q = y_q[0]

        
        return x_q, y_q

if __name__ == '__main__':
    pass