# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import string

n_hidden = 2048

all_letters = string.ascii_letters + ".,;"
n_letters = len(all_letters)

GT_KEYS = ['Andaman and Nicobar Islands', 'Andhra Pradesh', 
           'Arunachal Pradesh', 'Assam', 'Bihar', 
           'Chandigarh', 'Dadra and Nagar Haveli', 
           'Daman and Diu', 'Delhi', 'Goa', 'Gujarat', 
           'Haryana', 'Jharkhand', 
           'Jammu and Kashmir and Ladakh',
           'Karnataka', 'Kerala', 'Maharashtra', 
           'Manipur', 'Meghalaya', 'Mizoram', 'Madhya Pradesh',
           'Nagaland', 'Odisha', 'Puducherry', 'Punjab',
           'Rajasthan', 'Sikkim', 'Telangana', 'Tripura',
           'Uttar Pradesh', 'Uttarakhand']

def infer(net, name: str):
    net.eval()
    name_ohe = name_rep(name)
    hidden = net.init_hidden()

    for i in range(name_ohe.size()[0]):
        output, hidden = net(name_ohe[i], hidden)

    return output

def name_rep(name: str):
    rep = torch.zeros(len(name), 1, n_letters)
    for index, letter in enumerate(name):
        pos = all_letters.find(letter)
        rep[index][0][pos] = 1
    return rep

class GRU_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU_net, self).__init__()
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRU(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input_, hidden):
        out, hidden = self.gru_cell(input_.view(1, 1, -1), hidden)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output.view(1, -1), hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
