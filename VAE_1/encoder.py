import torch
import pandas as pd
import numpy as np
import os
import sys
from torch import nn
from torch.nn import functional as F
from itertools import combinations


class encoder(nn.Module):

    def __init__(
            self,
            device,
            domain_dims,
            init_emb_weights,
            z_dim=128,
            fc_layer_dims = (1024,512),
            dropout = 0.1
    ):
        super(encoder, self).__init__()
        self.device = device
        self.embedding_list = nn.ModuleList()
        self.num_domains = len(domain_dims)
        emb_dim = init_emb_weights[0].shape[1]
        self.emb_dim = emb_dim
        for i in range(self.num_domains):
            emb_i = nn.Embedding(
                    num_embeddings = domain_dims[i],
                    embedding_dim = emb_dim
            )
            emb_i.weight = torch.nn.Parameter(torch.from_numpy(init_emb_weights[i]),requires_grad=False)
            self.embedding_list.append(
                emb_i
            )

        self.FC_1 = nn.Linear(emb_dim*self.num_domains, fc_layer_dims[0])
        self.FC_mu = nn.Sequential(
            nn.Linear(fc_layer_dims[1],z_dim),
            nn.Tanh()
        )

        self.FC_sigma = nn.Sequential(
            nn.Linear(fc_layer_dims[1],z_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x_split = torch.chunk(x, self.num_domains, dim=-1)
        x_split = [_.squeeze(1) for _ in x_split]
        x_emb = []
        for i in range(self.num_domains):
            x_emb.append(self.embedding_list(x_split[i]))
        x_emb = torch.cat(x_emb, dim=-1)
        x1 = self.FC_1(x_emb)
        mu =  self.FC_mu(x1)
        log_var = self.FC_sigma(x1)
        return mu, log_var


