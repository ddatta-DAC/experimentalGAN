import torch
import pandas as pd
import numpy as np
import os
import sys
from torch import nn
from torch.nn import functional as F
from itertools import combinations


class decoder(nn.Module):

    def __init__(
            self,
            device,
            domain_dims,
            z_dim,
            gumbel_T=0.20,
            dropout = 0.1
    ):
        super(decoder, self).__init__()
        self.device = device
        self.z_dim = z_dim
        self.domain_dims = domain_dims  # e.g. [100,20,50,250]
        self.num_dims = len(self.domain_dims)

        self.gumbel_T = gumbel_T
        self.FC_List = nn.ModuleList()
        fc_inp_dim = self.z_dim
        # -------------------------
        # Each domain has its own projection (  softmax)
        # For each domain : FC ( z, z*2) , softmax()
        # -------------------------
        for i in range(self.num_dims):
            self.FC_List.append(
                nn.Sequential(
                    nn.Linear(self.z_dim, self.z_dim * 2),
                    nn.Dropout(dropout),
                    nn.Hardtanh(),
                    nn.Linear(self.z_dim * 2, self.domain_dims[i]),
                    nn.Dropout(dropout),
                    nn.LeakyReLU()
                )
            )
        self.mode = 'train'
        return

    def forward(
            self,
            z,
            return_01 = False
    ):
        x_generated = []
        for i in range(self.num_dims):
            # Concatenate z to lstm output
            _inp = z
            r = self.FC_List[i](_inp)
            # Apply softmax
            r = F.softmax(r, dim=-1)
            # Convert to indices
            if return_01 is False:
                r1 = torch.argmax(r, dim=1, keepdim=True)
                x_generated.append(r1)
            else:
                x_generated.append(r)
        # x2 has the indices
        x_generated = torch.cat(x_generated, dim=1)
        return x_generated

