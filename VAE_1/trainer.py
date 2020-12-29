from torch.utils.tensorboard import SummaryWriter
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch import optim
from tqdm import tqdm
import glob
import pandas as pd
import pickle
import time
from pathlib import Path
import multiprocessing as mp
import json
import yaml
from torch.nn import functional as F
from encoder import encoder
from decoder import decoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_embeddings(DIR):
    data_loc = './../generated_data_v1'
    domain_dims = get_domain_dims(data_loc, DIR)
    emb_weights = []
    for dom in sorted(domain_dims.keys()):
        file = sorted (glob.glob(os.path.join(data_loc, DIR, 'n2v_128_{}.npy'.format(dom))))
        emb = np.load(file)
        emb_weights.append(emb)
    emb_weights = np.array(emb_weights)
    return emb_weights


def get_domain_dims(data_loc, DIR):
    with open(os.path.join(data_loc, DIR, 'domain_dims.pkl'), 'rb') as fh:
        domain_dims = pickle.load(fh)
        return domain_dims

def read_data(DIR):
    ID_COL = 'PanjivaRecordID'
    data_loc = './../generated_data_v1'
    fname = 'train_data.csv'
    data_file = os.path.join(data_loc,DIR,fname)
    df = pd.read_csv(data_file, index_col=False)
    domain_dims = get_domain_dims(data_loc,DIR)
    try:
        del df[ID_COL]
    except:
        pass
    ordered_columns = list(sorted(domain_dims.keys()))
    df = df[ordered_columns]
    x = df.values
    return x,domain_dims

def vae_loss(x, x_r, mu, log_var ):

    reconst_loss = F.binary_cross_entropy(x_r, x, size_average=False)
    kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconst_loss, kl_div


class vae_module:
    def __init__(
            self,
            encoder_obj,
            decoder_obj,
            LR = 0.001
    ):
        self.encoder_obj = encoder_obj
        self.decoder_obj = decoder_obj
        self.LR = LR
        self.writer = SummaryWriter()
        self.optimizer = torch.optim.Adam([
            {'params': decoder_obj.parameters(), 'lr': self.LR * 0.5},
            {'params': encoder_obj.parameters(), 'lr': self.LR }]
        )
        self.num_steps = 0
        return

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std


    def train_model(self, data_x, batch_size = 125, num_epochs = 20, log_interval=100):
        num_batches = data_x.shape[0]//batch_size + 1
        idx = np.arange(data_x.shape[0])
        bs = batch_size

        for e in tqdm(range(num_epochs)):
            np.random.shuffle(idx)
            for b in range(num_batches):
                self.optimizer.zero_grad()
                self.num_steps += 1
                _idx = idx [b*bs:(b+1)*bs]
                x = data_x[_idx]

                mu, log_var = self.encoder_obj(x)
                z = self.reparameterize(mu, log_var)
                x_reconst = self.decoder_obj(z)
                r_loss, kld_loss =  vae_loss(x, x_reconst, mu, log_var)
                loss = r_loss + kld_loss
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar("Loss/reconstruction", r_loss, self.num_steps)
                self.writer.add_scalar("Loss/KLD", kld_loss, self.num_steps)

                if b % log_interval == 0:
                     print('Batch {} KLD loss {:.4f} Reconstruction loss {:.4f}'.format(b, kld_loss.data, r_loss.data))



DIR = 'us_import1'
X,domain_dims = read_data(DIR)
emb_weights = read_embeddings(DIR)
encoder_obj = encoder(
    device=DEVICE,
    domain_dims=domain_dims,
    init_emb_weights = emb_weights,
)

decoder_obj = decoder(
    device=DEVICE,
    domain_dims=domain_dims,
    init_emb_weights = emb_weights,
)

vae_module(encoder_obj, decoder_obj)


