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
    def pretrain_critic(self, num_epochs, data_loader, LR = None):
        if LR is None:
            LR = self.LR*2
            
        D_optimizer = optim.Adam(self.critic_obj.parameters(), lr=LR)
        print('DEVICE', self.device)
        for epoch in tqdm(range(num_epochs)):
            for i, data in enumerate(data_loader):
                D_optimizer.zero_grad()
                pos = data[0]
                pos = pos.to(self.device)
                neg = data[1]
                num_negSamples = neg.shape[1]
                neg = [ _.squeeze(1) for _ in torch.chunk(neg,num_negSamples,dim=1)]
                neg_loss = []
                for n in range(num_negSamples):
                    x_n = neg[n].to(self.device)
                    x_n_loss = self.critic_obj(x_n)
                    neg_loss.append(x_n_loss)
                    
                neg_loss = torch.cat(neg_loss, dim =-1)
                eps = 0.000001
                neg_loss = -torch.log( 1 - neg_loss + eps)
                neg_loss = torch.mean(neg_loss, dim=-1, keepdims=False)
                pos_loss = torch.log(self.critic_obj(pos))
                
                
                l2_reg = torch.norm(self.critic_obj.FC.weight) 
                _loss =  pos_loss + neg_loss  
                _loss = _loss.mean() + 0.01 * l2_reg
                _loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_obj.parameters(), 1)
                D_optimizer.step()
                self.dict_losses['pretrain_D'].append(_loss.cpu().data.numpy().mean())
                if i % self.log_interval == 0:
                    print('Pretrain  Epoch {}| Index {} loss {} '.format(epoch, i + 1, self.dict_losses['pretrain_D'][-1]))
                    
        return  self.dict_losses['pretrain_D']

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
        d_loss =  -d_real.mean() + (1 - d_generated).mean() + 0.01 * l2_reg 
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_obj.parameters(), 0.5)
        self.D_optimizer.step()
        self.dict_losses['D'].append(d_loss.cpu().data.numpy().mean())
        return

    def train(self, data_loader, data_loader_wNeg, num_epochs):
        self.num_steps = 0
        
        for epoch in tqdm(range(num_epochs)):
            
            if (epoch+1)%25 == 0:
                self.pretrain_critic(num_epochs = 2, data_loader = data_loader_wNeg , LR = self.LR/2)
                
            for i, data in enumerate(data_loader):
                data = data.to(self.device)
                self.num_steps += 1
                self._train_C(data)
                # Only update generator every |critic_iterations| iterations
                if (i+1) % self.critic_iterations == 0:
                    self._train_G(data)                    
                
                if (i+1) % self.log_interval == 0:
                    print("Iteration {} D:{:0.4f} G:{:0.4f} ".format(
                        self.num_steps, self.dict_losses['D'][-1], self.dict_losses['G'][-1])
                    )
                         
            
        return

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.shape[0]
        generated_data = Variable(generated_data.long(), requires_grad=True)
        prob_gen = self.critic_obj(generated_data)
        prob_real = self.critic_obj(real_data)
        
        print(prob_gen.shape, prob_real.shape )
        # Calculate gradients of probabilities with respect to examples
        gradients_g = torch_grad(
            outputs=prob_gen,
            inputs=generated_data,
            grad_outputs=torch.ones(prob_gen.size()).to(self.device),
            create_graph=True,
            retain_graph=True
        )
        
        print(gradients_g.shape)
        gradients_r = torch_grad(
            outputs=prob_real,
            inputs=generated_data,
            grad_outputs=torch.ones(real_data.size()).to(self.device),
            create_graph=True,
            retain_graph=True
        )

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



    def save_pretrained_D(self):
        PATH ='./pretrained_model.dat'
        torch.save(self.critic_obj.state_dict(), PATH)
        return
    
    def load_pretrained_D(self):
        PATH ='./pretrained_model.dat'
        self.critic_obj.load_state_dict(torch.load(PATH))
        return 
       