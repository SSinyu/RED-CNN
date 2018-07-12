### CUSTOM LOSS
# https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/


import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
from RED_CNN_dataloader import DCMsDataset
from RED_CNN_model import RED_CNN
from skimage.measure import compare_psnr, compare_ssim, compare_nrmse

class psnr_loss(nn.Module):
    def __init__(self):
        super(psnr_loss, self).__init__()
    def forward(self, img1, img2):
        if img1.size()[1] == 1:
            img1 = img1.squeeze(1)
        img1 = img1.data.cpu().numpy()
        if img2.size()[1] == 1:
            img2 = img2.squeeze(1)
        img2 = img2.data.cpu().numpy()
        loss_ = compare_psnr(img1, img2, data_range=4096)
        loss_tensor = torch.zeros(1,)
        loss_tensor[0,] = loss_
        loss = torch.sum(loss_tensor).to(device)
        return 1/loss*10000

class ssim_loss(nn.Module):
    def __init__(self):
        super(ssim_loss, self).__init__()
    def forward(self, img1, img2):
        if img1.size()[1] == 1:
            img1 = img1.squeeze(1)
        img1 = img1.data.cpu().numpy()
        if img2.size()[1] == 1:
            img2 = img2.squeeze(1)
        img2 = img2.data.cpu().numpy()
        loss_ = 1
        for i in range(img1.size()[0]):
            loss_ *= compare_ssim(img1[0], img2[0], data_range=4096)
        loss_tensor = torch.zeros(1,)
        loss_tensor[0,] = loss_
        loss = torch.sum(loss_tensor).to(device)
        return 1-loss

class nrmse_loss(nn.Module):
    def __init__(self):
        super(nrmse_loss, self).__init__()
    def forward(self, img1, img2):
        if img1.size()[1] == 1:
            img1 = img1.squeeze(1)
        img1 = img1.data.cpu().numpy()
        if img2.size()[1] == 1:
            img2 = img2.squeeze(1)
        img2 = img2.data.cpu().numpy()
        loss_ = compare_nrmse(img1, img2)
        loss_tensor = torch.zeros(1,)
        loss_tensor[0,] = loss_
        loss = torch.sum(loss_tensor).to(device)
        return 1-loss



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

red_cnn = RED_CNN(96).to(device)
criterion = ssim_loss()
optimizer = torch.optim.Adam(red_cnn.parameters(), lr=0.001)



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


total_step = len(dcmloader)
current_lr = LEARNING_RATE

for epoch in range(NUM_EPOCHS):
    for i, (inputs, targets) in enumerate(dcmloader):
        input_img = torch.tensor(inputs, requires_grad=True).unsqueeze(1).to(device)
        target_img = torch.tensor(targets).unsqueeze(1).to(device)
        outputs = red_cnn(input_img)
        loss = nn.MSELoss()(outputs, target_img) * 0 + criterion(outputs, target_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('EPOCH [{}/{}], STEP [{}/{}], LOSS {:.4f}'.format(epoch+1, NUM_EPOCHS, i+1, total_step, loss.item()))





torch.save(red_cnn.state_dict(), 'redcnn_50ep(psnr_loss).ckpt')


