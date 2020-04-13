import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class RNNDataset(Dataset):
    def __init__(self, data, window_size=12):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) // self.window_size
  
    def __getitem__(self, idx):
        idx = idx + self.window_size
        data = torch.zeros(self.window_size, self.data.shape[1])
        target = torch.zeros(self.window_size, self.data.shape[1])
        for i in range(0, self.window_size):
            data[i] = torch.tensor(self.data[idx+i])
            target[i] = data[i]
            
        return data, target