import torch
import math
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd


class CatDataset_W_Neg(torch.utils.data.Dataset):
    def __init__(self, data_pos_path, data_neg_path):
        super(CatDataset_W_Neg).__init__()
        self.data_pos = pd.read_csv(data_pos_path, index=None).values
        self.data_neg = pd.read_csv(data_neg_path, index=None).values

    def __getitem__(self, index) :
        return self.data_pos[index], self.data_neg[index]

    def __len__(self):
        return self.data_pos.shape[0]


class CatDataset(torch.utils.data.Dataset):
    def __init__(self, data_pos_path, data_neg_path):
        super(CatDataset_W_Neg).__init__()
        self.data = pd.read_csv(data_pos_path, index=None)

    def __getitem__(self, index) :
        return  self.data_pos[index]

    def __len__(self):
        return self.data.shape[0]

# -----------------------------------------------
# data = np.random.random([500,10])
# ds = CatDataset_W_Neg(data)
# DL = DataLoader(
#     ds,
#     batch_size=10,
#     shuffle=False,
#     num_workers=5,
#     pin_memory=True
# )
# for idx,x in enumerate(DL):
#     print(idx, x[0].shape,x[1].shape)
