import torch
import numpy as np
from torch.nn import Sequential, Linear, ReLU, Module, Tanh
from torch.nn.functional import mse_loss


class SimpleAutoEncoder(torch.nn.Module):
    def __init__(self, num_inputs, val_lambda=666):
        super(SimpleAutoEncoder, self).__init__()
        self.val_lambda = val_lambda
        self.neurons_l1_to_l2 = 12
        self.neurons_l2_to_latent = 8
        
        self.encoder = Sequential(
            Linear(num_inputs, self.neurons_l1_to_l2),
            ReLU(True),
            Linear(self.neurons_l1_to_l2, self.neurons_l2_to_latent),
            Tanh()
        )
        
        self.decoder = Sequential(
            Linear(self.neurons_l2_to_latent,self.neurons_l1_to_l2),
            ReLU(True),
            Linear(self.neurons_l1_to_l2, num_inputs),
            Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def set_lambda(self, val_lambda):
        self.val_lambda = val_lambda
        print('Set lambda of model to: {}'.format(self.val_lambda))
    
    def calc_reconstruction_error(self,x):
        re = mse_loss(self.forward(x),x)
        return re
    
    def predict_binary(self,x):
        re = self.calc_reconstruction_error(x)
        if re.data.item() > self.val_lambda:
            return 1
        else:
            return 0

    @staticmethod
    def weight_init(m):
        if isinstance(m, torch.nn.Linear):
            size = m.weight.size()
            fan_out = size[0]  # number of rows
            fan_in = size[1]  # number of columns
            variance = np.sqrt(2.0 / (fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)
