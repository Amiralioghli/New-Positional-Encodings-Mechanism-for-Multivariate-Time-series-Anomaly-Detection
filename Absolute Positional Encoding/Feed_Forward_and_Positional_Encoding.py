from torch import nn
import torch
import math
import numpy as np
device = torch.device("mps")

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class Absolut_Positional_Encoding(nn.Module):
    def __init__(self, d_model, max_len=32, dim=3, device=device):
        super(Absolut_Positional_Encoding, self).__init__()
        self.device = device
        self.dim = dim

        angle_rads = self.get_angles(torch.arange(max_len).unsqueeze(1),
                                      torch.arange(d_model).unsqueeze(0),
                                      d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        if dim == 3:
            self.pos_encoding = angle_rads.unsqueeze(0)
        elif dim == 4:
            self.pos_encoding = angle_rads.unsqueeze(0).unsqueeze(0)
        self.pos_encoding = self.pos_encoding.to(device=self.device)

    def forward(self, x):
        return x + self.pos_encoding[:, :x.size(1)]

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
        return pos * angle_rates

