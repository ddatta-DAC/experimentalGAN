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
            z_dim=128,
            init_emb_weights=None,
            dropout = 0.1
    ):
        super(decoder, self).__init__()
        self.device = device
        self.z_dim = z_dim
        self.domain_dims = domain_dims  # e.g. [100,20,50,250]
        self.num_dims = len(self.domain_dims)

        self.FC_List = nn.ModuleList()
        fc_inp_dim = self.z_dim
        # -------------------------
        # Each domain has its own projection (  softmax)
        # For each domain : FC ( z, z*2) , softmax()
        # -------------------------
        emb_dim = init_emb_weights.shape[-1]
        for i in range(self.num_dims):

            _layers = []
            _layers.append(  nn.Linear(self.z_dim, emb_dim ))
            _layers.append(nn.Dropout(dropout))
            _layers.append(nn.LeakyReLU())
            _w = init_emb_weights[i].transpose()
            _linear = nn.Linear(emb_dim, self.domain_dims[i], bias=False)
            _linear.weight = torch.nn.Parameter(torch.from_numpy(_w), requires_grad=False)
            _layers.append(_linear)
            _layers.append(nn.Softmax(dim=-1))
            self.FC_List.append(
                nn.Sequential(*_layers)
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

            # Convert to indices
            if return_01 is False:
                r1 = torch.argmax(r, dim=1, keepdim=True)
                x_generated.append(r1)
            else:
                x_generated.append(r)
        # x2 has the indices
        x_generated = torch.cat(x_generated, dim=1)
        return x_generated

