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
            critic_iterations=5,
            log_interval=100,
            LR=0.0001,
            model_save_dir = 'saved_model'
            log_dir = 'log_dir'
    ):
        self.device = device
        self.LR = LR
        self.ts = str(time.time()).split('.')[0]
        self.eps = 0.000001
        self.model_signature = '_'.join([str(generator_obj.z_dim), str(critic_obj.emb_dim)], self.ts )
        self.save_dir = model_save_dir
        pathobj = Path(self.save_dir)
        pathobj.mkdir(exist_ok=True, parents=True)
        
        self.log_dir = os.path.join(log_dir,  self.model_signature )
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
        return
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
                
                neg_loss = -torch.log( 1 - neg_loss + self.eps)
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

    

    
    def save_pretrained_D(self):
        PATH = os.path.join(self.save_dir.'model_D_{}.dat'.format(self.model_signature))
        torch.save(self.critic_obj.state_dict(), PATH)
        return
    
    # Get the latest file
    def load_pretrained_D(self):
        PATH ='./pretrained_model.dat'
        self.critic_obj.load_state_dict(torch.load(PATH))
        return 
    
    
    def save_pretrained_G(self):
        PATH = os.path.join(self.save_dir.'model_G_{}.dat'.format(self.model_signature))
        torch.save(self.generator_obj.state_dict(), PATH)
        return
    
    # Get the latest file
    def load_pretrained_G(self):
        PATH ='./pretrained_model_G.dat'
        self.generator_obj.load_state_dict(torch.load(PATH))
        return 
       
    def gen_test_samples(self, num_samples):
        self.generator_obj.eval()
        generated_data = self.generator_obj(num_samples)
        self.generator_obj.train()
        return generated_data.cpu().data.numpy()
    