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
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(
            self,
            device,
            generator_obj,
            critic_obj_1,
            critic_obj_2,
            critic_iterations=5,
            log_interval=100,
            LR=0.0001,
            model_save_dir='saved_model',
            log_dir='log_dir'
    ):
        self.device = device
        self.LR = LR
        self.ts = str(time.time()).split('.')[0]
        self.eps = 0.0000001
        self.model_signature = '_'.join([str(generator_obj.z_dim), str(critic_obj.emb_dim), self.ts])

        self.save_dir = model_save_dir
        pathobj = Path(self.save_dir)
        pathobj.mkdir(exist_ok=True, parents=True)

        self.log_dir = os.path.join(log_dir, self.model_signature)
        pathobj = Path(self.log_dir)
        pathobj.mkdir(exist_ok=True, parents=True)
        self.GP = 2
        self.G_optimizer = optim.Adam(generator_obj.parameters(), lr=self.LR )
        self.D_optimizer_1 = optim.RMSprop(critic_obj_1.parameters(), lr=self.LR )
        self.D_optimizer_2 = optim.RMSprop(critic_obj_2.parameters(), lr=self.LR )
        self.log_interval = log_interval
        self.critic_iterations = critic_iterations
        self.generator_obj = generator_obj
        self.critic_obj_1 = critic_obj_1
        self.critic_obj_2 = critic_obj_2
        self.dict_losses = {'D': [], 'G': [] }
        self.writer = SummaryWriter()
        self.num_steps = 0
        return

    def write_out_log(self):
        pathobj = Path(self.log_dir)
        pathobj.mkdir(exist_ok=True, parents=True)
        log_file = os.path.join(self.log_dir, 'losses.yaml')
        if not os.path.exists(log_file):
            print(' Creating log file ...')
        data = self.dict_losses
        with open(log_file, 'w') as fh:
            yaml.dump(data, fh)
        return


    def _train_G(self, data, ):
        self.G_optimizer.zero_grad()
        batch_size = data.shape[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        _rnd = np.random.uniform()
        if _rnd < 0.5:
            gen_loss = self.critic_obj_1(generated_data)
        else:
            gen_loss = self.critic_obj_2(generated_data)

        # Use the formulation of LSGAN
        c = 1.0
        g_loss = 0.5 * torch.square(torch.mean(gen_loss) - c)
        g_loss.backward()

        torch.nn.utils.clip_grad_value_(self.generator_obj.parameters(), 1)
        self.G_optimizer.step()
        # Record loss
        self.dict_losses['G'].append(g_loss.cpu().data.numpy().mean())
        self.writer.add_scalar("Loss/train_G", g_loss, self.num_steps)
        self.writer.flush()
        return

    def sample_generator(self, num_samples):
        generated_data = self.generator_obj(num_samples)
        return generated_data

    def _train_C(self, data):
        self.D_optimizer_1.zero_grad()
        self.D_optimizer_2.zero_grad()
        batch_size = data.shape[0]
        generated_data = self.sample_generator(batch_size)
        # Calculate probabilities on real and generated data
        _rnd = np.random.uniform()
        if _rnd < 0.5:
            d_real = torch.sigmoid(self.critic_obj_1(data))
            d_generated = torch.sigmoid(self.critic_obj_1(generated_data))
            GP = self.gradient_penalty(data)

            # Create total loss and optimize
            # Maximize d_real, minimize d_generated

            d_loss = - (torch.log(d_real.mean() + self.eps) - torch.log(d_generated.mean() + self.eps)) + GP
            d_loss.backward()
            torch.nn.utils.clip_grad_value_(self.critic_obj.parameters(), 1)
            self.D_optimizer_1.step()
            self.critic_obj_1.normalize_embedding()
        if _rnd >= 0.5:
            d_real = torch.sigmoid(self.critic_obj_2(data))
            d_generated = torch.sigmoid(self.critic_obj_2(generated_data))
            GP = self.gradient_penalty(data)

            # Create total loss and optimize
            # Maximize d_real, minimize d_generated

            d_loss = - (torch.log(d_real.mean() + self.eps) - torch.log(d_generated.mean() + self.eps)) + GP
            d_loss.backward()
            torch.nn.utils.clip_grad_value_(self.critic_obj_2.parameters(), 1)
            self.D_optimizer_2.step()
            self.critic_obj_2.normalize_embedding()


        self.dict_losses['D'].append(d_loss.cpu().data.numpy().mean())
        self.writer.add_scalar("Loss/train_D", d_loss, self.num_steps)
        self.writer.flush()
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

    # Calculate gradient penalty using real data
    def gradient_penalty(self, real_data, _critic_obj_):
        i1 = real_data.detach().to(self.device)
        i1.requires_grad_(True)
        critic_i1 = _critic_obj_(i1)

        gradients_1 = torch_grad(
            outputs=critic_i1,
            inputs=i1,
            grad_outputs=torch.ones(critic_i1.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients_1.view(gradients_1.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.GP
        return gradient_penalty

    def gen_test_samples(self, num_samples):
        self.generator_obj.eval()
        generated_data = self.generator_obj(num_samples)
        self.generator_obj.train()
        return generated_data.cpu().data.numpy()
