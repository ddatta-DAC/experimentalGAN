import pandas as pd
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


class Trainer():
    def __init__(
            self,
            device,
            generator_obj,
            critic_obj,
            GP_weight=10,
            critic_iterations=5,
            log_interval =100,
        ):
        self.device = device

    def pretrain_critic(self):
        return

    def _train_G(self):
        return
    def _train_C(self):
        return

    def train(self, data_loader, num_epochs):
        return

    def _gradient_penalty(self, real_data, generated_data):
       return

