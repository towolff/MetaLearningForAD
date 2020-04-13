import torch
from torch.nn import Sequential, Linear, ReLU, Module, Tanh
from torch.autograd import Variable
from torch.nn.functional import mse_loss
import numpy as np


class EncoderLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_space_size, num_layers=1, isCuda=False):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_space_size = latent_space_size
        self.isCuda = isCuda
        
        self.lstm_1 = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.relu = torch.nn.ReLU()
        self.lstm_2 = torch.nn.LSTM(input_size=self.hidden_size, hidden_size=self.latent_space_size, num_layers=self.num_layers, batch_first=True)
        self.tanh = torch.nn.Tanh()
        
        #initialize weights
        torch.nn.init.xavier_uniform(self.lstm_1.weight_ih_l0, gain=np.sqrt(2))
        torch.nn.init.xavier_uniform(self.lstm_1.weight_hh_l0, gain=np.sqrt(2))
        
        torch.nn.init.xavier_uniform(self.lstm_2.weight_ih_l0, gain=np.sqrt(2))
        torch.nn.init.xavier_uniform(self.lstm_2.weight_hh_l0, gain=np.sqrt(2))
        
        
        
    def forward(self, input):
        tt = torch.cuda if self.isCuda else torch
        
        h0_1 = Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size))
        c0_1 = Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size))
        encoded_input, hidden = self.lstm_1(input, (h0_1, c0_1))
        encoded_input = self.relu(encoded_input)

        h0_2 = Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.latent_space_size))
        c0_2 = Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.latent_space_size))
        encoded_input, hidden = self.lstm_2(encoded_input, (h0_2, c0_2))
        encoded_input = self.tanh(encoded_input)
        
        return encoded_input


class DecoderLSTM(torch.nn.Module):
    def __init__(self, hidden_size, output_size, latent_space_size, num_layers=1, isCuda=False):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.latent_space_size = latent_space_size
        self.isCuda = isCuda
        
        self.lstm_1 = torch.nn.LSTM(self.latent_space_size, self.hidden_size, self.num_layers, batch_first=True)
        self.relu = torch.nn.ReLU()
        self.lstm_2 = torch.nn.LSTM(self.hidden_size, self.output_size, self.num_layers, batch_first=True)
        self.tanh = torch.nn.Tanh()
        
        #initialize weights
        torch.nn.init.xavier_uniform(self.lstm_1.weight_ih_l0, gain=np.sqrt(2))
        torch.nn.init.xavier_uniform(self.lstm_1.weight_hh_l0, gain=np.sqrt(2))
        
        torch.nn.init.xavier_uniform(self.lstm_2.weight_ih_l0, gain=np.sqrt(2))
        torch.nn.init.xavier_uniform(self.lstm_2.weight_hh_l0, gain=np.sqrt(2))
        
    def forward(self, encoded_input):
        tt = torch.cuda if self.isCuda else torch
        
        h0_1 = Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.hidden_size))
        c0_1 = Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.hidden_size))
        decoded_output, hidden = self.lstm_1(encoded_input, (h0_1, c0_1))
        decoded_output = self.relu(decoded_output)
        
        h0_2 = Variable(tt.FloatTensor(self.num_layers, decoded_output.size(0), self.output_size))
        c0_2 = Variable(tt.FloatTensor(self.num_layers, decoded_output.size(0), self.output_size))
        decoded_output, hidden = self.lstm_2(decoded_output, (h0_2, c0_2))
        decoded_output = self.tanh(decoded_output)
        
        return decoded_output


class LSTMAutoEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_space_size, num_layers=1, isCuda=False):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, latent_space_size, num_layers, isCuda)
        self.decoder = DecoderLSTM(hidden_size, input_size, latent_space_size, num_layers, isCuda)
        
    def forward(self, input):
        encoded_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)
        return decoded_output
    
    def calc_reconstruction_error(self,x):
        epsilon = 10**-8
        re = mse_loss(self.forward(x),x) + epsilon
        return re