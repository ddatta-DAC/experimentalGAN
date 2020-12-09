import torch
import pandas as pd
import numpy as np
import os
import sys
from torch import nn
from torch.nn import functional as F
from itertools import combinations

class generator_v1(nn.Module):

    def __init__ (self, device, domain_dims, z_dim, lstm_hidden_dims=256, lstm_num_layers = 2, gumbel_T= 0.25):
        super(generator_v1, self).__init__()
        self.device = device
        self.z_dim = z_dim
        self.domain_dims = domain_dims # e.g. [100,20,50,250]
        self.num_dims = len(self.domain_dims)
        self.lstm_hidden_dims = lstm_hidden_dims

        self.lstm = nn.LSTM(
            input_size = self.z_dim + self.num_dims,
            hidden_size = lstm_hidden_dims,
            num_layers = lstm_num_layers,
            batch_first = True,
            bidirectional = True
        )

        self.gumbel_T = gumbel_T
        self.FC_List = nn.ModuleList()
        fc_inp_dim = self.z_dim + self.lstm_hidden_dims*2 # Append z to each of the outputs of the LSTM
        for i in range(self.num_dims):
            self.FC_List.append(
                    nn.Sequential(
                    nn.BatchNorm1d(fc_inp_dim),
                    nn.Linear(fc_inp_dim, self.domain_dims[i] ),
                    nn.Dropout(0.1),
                    nn.LeakyReLU()
                )
            )


    def forward(self, sample_size = 1):
        # Random sample
        z = torch.normal( mean=0, std = 1, size=[sample_size, self.z_dim]).to(self.device)
        
        # Append position vector to each input
        pos = torch.eye(self.num_dims).reshape([self.num_dims * self.num_dims]).repeat(sample_size,1).view(-1, self.num_dims , self.num_dims ).to(self.device)
        z1 = z.repeat(1, self.num_dims).reshape([-1, self.num_dims, self.z_dim  ]) # Shape : [ batch, self.num_dims , self.z_dim]

        # lstm input
        lstm_input = torch.cat([z1, pos],dim=-1) # Shape [batch, self.num_dims , self.z_dim + self.num_dims ]
        # Output shape : (batch, seq, feature)
        lstm_op, _  = self.lstm(lstm_input)

        # Split the lstm_op
        x1 = torch.chunk(lstm_op, self.num_dims, dim=1)
        x1 = [ _.squeeze(1) for _ in x1]
        x2 = []

        for i in range(self.num_dims):           
            # Concatenate z to lstm output
            _inp = torch.cat( [x1[i],z], dim=1)
            r = self.FC_List[i](_inp)
            # Apply gumbel softmax
            r = F.gumbel_softmax(r, tau=self.gumbel_T, hard=True)
            r1 = torch.argmax(r, dim=1, keepdim=True)
            x2.append(r1)

        # x2 has the indices
        x2 = torch.cat(x2, dim=1)
        return x2


# obj = generator_v1 (domain_dims=[100,150,300], z_dim=20, lstm_hidden_dims=256, lstm_num_layers=2)
# a = obj.forward(40)
# print(a)