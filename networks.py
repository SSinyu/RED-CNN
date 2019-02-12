import os
import numpy as np
import torch.nn as nn

class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv_first = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t_last = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x.clone()
        out = self.relu(self.conv_first(x))
        out = self.relu(self.conv(out))
        residual_2 = out.clone()
        out = self.relu(self.conv(out))
        out = self.relu(self.conv(out))
        residual_3 = out.clone()
        out = self.relu(self.conv(out))

        # decoder
        out = self.conv_t(out)
        out += residual_3
        out = self.conv_t(self.relu(out))
        out = self.conv_t(self.relu(out))
        out += residual_2
        out = self.conv_t(self.relu(out))
        out = self.conv_t_last(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out
