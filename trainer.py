import pandas as pd
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch import optim
from tqdm import tqdm


class Trainer():
    def __init__(
            self,
            device,
            generator_obj,
            critic_obj,
            GP_weight=10,
            critic_iterations=5,
            log_interval=100,
            LR=0.0001
    ):
        self.device = device
        self.LR = LR
        self.GP_weight = GP_weight

        self.G_optimizer = optim.Adam(generator_obj.parameters(), lr=self.LR)
        self.D_optimizer = optim.Adam(critic_obj.parameters(), lr=self.LR)
        self.log_interval = log_interval
        self.critic_iterations = critic_iterations
        self.generator_obj = generator_obj
        self.critic_obj = critic_obj
        self.dict_losses = {}
        self.dict_losses['gradient_norm'] = []
        self.dict_losses['GP'] = []
        self.dict_losses['D'] = []
        self.dict_losses['G'] = []
        self.dict_losses['pretrain_D'] = []

    # -----------------
    # Pretrain the critic using negative sample instances
    # -----------------
    def pretrain_critic(self, num_epochs, data_loader):
        for epoch in tqdm(range(num_epochs)):
            for i, data in enumerate(data_loader):
                self.D_optimizer.zero_grad()
                pos = data[0]
                neg = data[1]
                pos_loss = self.critic_obj(pos)
                neg_loss = self.critic_obj(neg)
                _loss = pos_loss - neg_loss
                _loss = _loss.mean()
                _loss.backward()

                self.D_optimizer.step()
                self.dict_losses['pretrain_D'].append(_loss.cpu().data.numpy().mean())
                if i % self.log_interval == 0:
                    print('Pretrain  Epoch {}| Index {} loss '.format(epoch, i + 1, self.dict_losses['pretrain_D'][-1]))
        return

    def _train_G(self, data):
        self.G_optimizer.zero_grad()
        batch_size = data.shape[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        gen_loss = self.critic_obj(generated_data)
        g_loss = - gen_loss.mean()
        g_loss.backward()
        self.G_optimizer.step()
        # Record loss
        self.dict_losses['G'].append(g_loss.data[0])
        return

    def sample_generator(self, num_samples):
        generated_data = self.generator_obj(num_samples)
        return generated_data

    def _train_C(self, data):
        self.D_optimizer.zero_grad()
        batch_size = data.shape[0]
        generated_data = self.sample_generator(batch_size)
        # Calculate probabilities on real and generated data
        data = Variable(data)
        d_real = self.critic_obj(data)
        d_generated = self.critic_obj(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.dict_losses['GP'].append(gradient_penalty.cpu().data.numpy().mean())

        # Create total loss and optimize
        self.D_optimizer.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()
        self.D_optimizer.step()
        return

    def train(self, data_loader, num_epochs):
        self.num_steps = 0
        for epoch in tqdm(range(num_epochs)):
            print("\nEpoch {}".format(epoch + 1))
            for i, data in enumerate(data_loader):
                self.num_steps += 1
                self._train_C(data)
                # Only update generator every |critic_iterations| iterations
                if self.num_steps % self.critic_iterations == 0:
                    self._train_G(data)
                if i % self.log_interval == 0:
                    print("Iteration {:0.4f} D: G:{:0.4f} GP:{:0.4f} Gradient norm: {:0.4f}".format(
                        i + 1, self.dict_losses['D'][-1], self.losses['G'][-1], self.losses['GP'][-1],
                        self.losses['gradient_norm'][-1])
                    )
        return

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.shape[0]
        prob_gen = self.critic_obj(generated_data)
        prob_real = self.critic_obj(real_data)

        # Calculate gradients of probabilities with respect to examples
        gradients_g = torch_grad(
            outputs=prob_gen,
            inputs=generated_data,
            grad_outputs=torch.ones(prob_gen.size()).to(self.device),
            create_graph=True,
            retain_graph=True
        )[0]
        gradients_r = torch_grad(
            outputs=prob_real,
            inputs=generated_data,
            grad_outputs=torch.ones(real_data.size()).to(self.device),
            create_graph=True,
            retain_graph=True
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        approx_grad = (gradients_r + gradients_g) / 2
        approx_grad = approx_grad.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(approx_grad ** 2, dim=1) + 1e-12)
        self.dict_losses['gradient_norm'].append(gradients_norm.cpu().data.numpy())
        # Return gradient penalty
        return self.GP_weight * ((gradients_norm - 1) ** 2).mean()



