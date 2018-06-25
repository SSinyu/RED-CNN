
import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
import cv2
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from logger import Logger
from torch.utils.data import DataLoader
from RED_CNN_dataloader import DCMsDataset


# RED-CNN
class RED_CNN(nn.Module):
    def __init__(self, out_channels=96):
        super(RED_CNN, self).__init__()
        self.conv_first = nn.Conv2d(1, out_channels, kernel_size=5, stride=1, padding=0)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=0)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=5, stride=1, padding=0)
        self.deconv_last = nn.ConvTranspose2d(out_channels, 1, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        residual1 = x # 28*28
        layer01 = self.conv_first(x)
        layer01 = self.relu(layer01)
        residual2 = layer01 # 24*24
        layer02 = self.conv(layer01)
        layer02 = self.relu(layer02)
        residual3 = layer02 # 20*20
        layer03 = self.conv(layer02)
        layer03 = self.relu(layer03)
        residual4 = layer03 # 16*16
        layer04 = self.conv(layer03)
        layer04 = self.relu(layer04)
        residual5 = layer04 # 12*12
        layer05 = self.conv(layer04)
        layer05 = self.relu(layer05) # 8*8
        # Decoder
        layer06 = self.deconv(layer05) # 12*12
        layer06 += residual5
        layer06 = self.relu(layer06)
        layer07 = self.deconv(layer06) # 16*16
        layer07 += residual4
        layer07 = self.relu(layer07)
        layer08 = self.deconv(layer07) # 20*20
        layer08 += residual3
        layer08 = self.relu(layer08)
        layer09 = self.deconv(layer08) # 24*24
        layer09 += residual2
        layer09 = self.relu(layer09)
        layer10 = self.deconv_last(layer09) # 28*28
        layer10 += residual1
        layer10 = self.relu(layer10)
        return layer10


'''
LEARNING_RATE = 1e-3
LEARNING_RATE_ = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 32
OUT_CHANNELS = 96

# patch datasets
input_img_dir = '/home/datascience/PycharmProjects/CT/patch/input/'
target_img_dir = '/home/datascience/PycharmProjects/CT/patch/target/'
#print(os.listdir(input_img_dir)[:3])
#print(os.listdir(target_img_dir)[:3])

dcm = DCMsDataset(input_img_dir, target_img_dir)
dcmloader = DataLoader(dcm, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dcm, batch_size=BATCH_SIZE, shuffle=False)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
redcnn = RED_CNN(OUT_CHANNELS).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(redcnn.parameters(), lr=LEARNING_RATE)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

total_step = len(dcmloader)
current_lr = LEARNING_RATE

for epoch in range(NUM_EPOCHS):
    for i, (inputs, targets) in enumerate(dcmloader):
        input_img = torch.tensor(inputs, requires_grad=True).unsqueeze(1).to(device)
        target_img = torch.tensor(targets).unsqueeze(1).to(device)
        outputs = redcnn(input_img)
        loss = criterion(outputs, target_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('EPOCH [{}/{}], STEP [{}/{}], LOSS {:.4f}'.format(epoch+1, NUM_EPOCHS, i+1, total_step, loss.item()))

        if (epoch+1) % 10 == 0:
            current_lr /= 1.2
            update_lr(optimizer, current_lr)



torch.save(redcnn.state_dict(), 'redcnn_100ep.ckpt')


'''


'''
redcnn.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for input_img, target_img in test_loader:
        input_img = input_img.to(device)
        target_img = target_img.to(device)
'''







'''
# MNIST dataset
transform = transforms.Compose([
    transforms.Resize((8,8)),
    transforms.Resize((28,28)),
    transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root='../../data/', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)
target_dataset = torchvision.datasets.MNIST(root='../../data/', train=True, transform=transforms.ToTensor(), download=True)
target_loader = torch.utils.data.DataLoader(dataset=target_dataset, batch_size=64, shuffle=False)
'''