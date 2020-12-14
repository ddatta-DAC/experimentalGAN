import pandas as pd
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
import time
from pathlib import Path
import multiprocessing as mp
import json
import yaml

class Trainer():
    def __init__(
            self,
            device,
            generator_obj,
            critic_obj,
            critic_iterations=5,
            log_interval=100,
            LR=0.0001,
            replay_interval = 40,
            model_save_dir='saved_model',
            log_dir='log_dir'
    ):
        self.device = device
        self.LR = LR
        self.ts = str(time.time()).split('.')[0]
        self.eps = 0.000001
        self.replay_interval = replay_interval
        self.model_signature = '_'.join([str(generator_obj.z_dim), str(critic_obj.emb_dim), self.ts])
        self.save_dir = model_save_dir
        pathobj = Path(self.save_dir)
        pathobj.mkdir(exist_ok=True, parents=True)

        self.log_dir = os.path.join(log_dir, self.model_signature)
        pathobj = Path(self.save_dir)
        pathobj.mkdir(exist_ok=True, parents=True)
        self.G_optimizer = optim.Adam(generator_obj.parameters(), lr=self.LR)
        self.D_optimizer = optim.Adam(critic_obj.parameters(), lr=self.LR)
        self.log_interval = log_interval
        self.critic_iterations = critic_iterations
        self.generator_obj = generator_obj
        self.critic_obj = critic_obj
        self.dict_losses = {}
        self.dict_losses['D'] = []
        self.dict_losses['G'] = []
        self.dict_losses['pretrain_D'] = []

    def write_out_log(self):
        log_file = os.path.join(self.log_dir, 'losses.yaml')
        if not os.path.exists(log_file):
            print(' Creating log file ...')
        data = self.dict_losses
        with open(log_file, 'w') as fh:
            yaml.dump(data, fh)
        return

    # -----------------
    # Pretrain the critic using negative sample instances
    # The loss function :
    # Maximize:
    # log( F(X_p)) / Mean(1 -F(X_n)) )
    # = log(F(X_p)) - Mean log(1 -F(X_n))
    # -----------------
    def pretrain_critic(self, num_epochs, data_loader, LR=None):
        if LR is None:
            LR = self.LR * 2

        D_optimizer = optim.Adam(self.critic_obj.parameters(), lr=LR)
        print('DEVICE', self.device)
        for epoch in tqdm(range(num_epochs)):
            for i, data in enumerate(data_loader):
                D_optimizer.zero_grad()
                pos = data[0]
                pos = pos.to(self.device)
                neg = data[1]
                num_negSamples = neg.shape[1]
                neg = [_.squeeze(1) for _ in torch.chunk(neg, num_negSamples, dim=1)]
                neg_x_res = []

                for n in range(num_negSamples):
                    x_n = neg[n].to(self.device)
                    x_n_loss = self.critic_obj(x_n)
                    neg_x_res.append(x_n_loss)
                # --------------------
                # Loss function
                # --------------------
                neg_x_res = torch.cat(neg_x_res, dim=-1)  # shape [batch, num_neg_samples]
                # Mean(log(1 - f(x_n))
                neg_loss = torch.log((1 - neg_x_res) + self.eps)
                neg_loss = torch.mean(neg_loss, dim=-1, keepdims=False)

                # log (f(x_p))
                pos_loss = torch.log(self.critic_obj(pos))
                l2_reg = torch.norm(self.critic_obj.FC.weight)
                _loss = pos_loss + neg_loss
                _loss = - _loss.mean() + 0.001 * l2_reg
                _loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_obj.parameters(), 1)
                D_optimizer.step()
                self.dict_losses['pretrain_D'].append(_loss.cpu().data.numpy().mean())
                if i % self.log_interval == 0:
                    print('Pretrain  Epoch {}| Index {} loss {} '.format(epoch, i + 1,
                                                                         self.dict_losses['pretrain_D'][-1]))

        return self.dict_losses['pretrain_D']

    def _train_G(self, data):
        self.G_optimizer.zero_grad()
        batch_size = data.shape[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        gen_loss = self.critic_obj(generated_data)
        g_loss = (1 - gen_loss)
        g_loss = g_loss.mean()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator_obj.parameters(), 1)
        self.G_optimizer.step()
        # Record loss
        self.dict_losses['G'].append(g_loss.cpu().data.numpy().mean())
        return

    def sample_generator(self, num_samples):
        generated_data = self.generator_obj(num_samples)
        return generated_data

    def _train_C(self, data):
        self.D_optimizer.zero_grad()
        batch_size = data.shape[0]
        generated_data = self.sample_generator(batch_size)
        # Calculate probabilities on real and generated data

        d_real = self.critic_obj(data)
        d_generated = self.critic_obj(generated_data)
        l2_reg = torch.norm(self.critic_obj.FC.weight)

        # Create total loss and optimize
        # Maximize d_real, minimize d_generated
        d_loss = -d_real.mean() + (1 - d_generated).mean() + 0.01 * l2_reg
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_obj.parameters(), 0.5)
        self.D_optimizer.step()
        self.dict_losses['D'].append(d_loss.cpu().data.numpy().mean())
        return

    def train(self, data_loader, data_loader_wNeg, num_epochs):
        self.num_steps = 0
        for epoch in tqdm(range(num_epochs)):
            if (epoch + 1) % self.replay_interval == 0:
                self.pretrain_critic(num_epochs=2, data_loader=data_loader_wNeg, LR=self.LR / 2)
            for i, data in enumerate(data_loader):
                data = data.to(self.device)
                self.num_steps += 1
                self._train_C(data)
                # Only update generator every |critic_iterations| iterations
                if (i + 1) % self.critic_iterations == 0:
                    self._train_G(data)

                if (i + 1) % self.log_interval == 0:
                    print("Iteration {} D:{:0.4f} G:{:0.4f} ".format(
                        self.num_steps, self.dict_losses['D'][-1], self.dict_losses['G'][-1])
                    )
        self.write_out_log()
        return

    def save_pretrained_D(self):
        PATH = os.path.join(self.save_dir, 'model_D_{}.dat'.format(self.model_signature))
        torch.save(self.critic_obj.state_dict(), PATH)
        return

    # Get the latest file
    def load_pretrained_D(self):
        # check for the latest file
        _path_pattern = 'model_D_' + '_'.join([str(self.generator_obj.z_dim), str(self.critic_obj.emb_dim)]) + '**.dat'
        _path_pattern = os.path.join(self.save_dir, _path_pattern)
        try:
            PATH = sorted(glob.glob(_path_pattern))[-1]
            self.critic_obj.load_state_dict(torch.load(PATH))
        except:
            print('ERROR no file found!')
        return

    def save_pretrained_G(self):
        PATH = os.path.join(self.save_dir, 'model_G_{}.dat'.format(self.model_signature))
        torch.save(self.generator_obj.state_dict(), PATH)
        return

    # Get the latest file
    def load_pretrained_G(self):
        _path_pattern = 'model_G_' + '_'.join([str(self.generator_obj.z_dim), str(self.critic_obj.emb_dim)]) + '**.dat'
        _path_pattern = os.path.join(self.save_dir, _path_pattern)
        try:
            PATH = sorted(glob.glob(_path_pattern))[-1]
            self.critic_obj.load_state_dict(torch.load(PATH))
        except:
            print('ERROR no file found!')

    def gen_test_samples(self, num_samples):
        self.generator_obj.eval()
        generated_data = self.generator_obj(num_samples)
        self.generator_obj.train()
        return generated_data.cpu().data.numpy()
