
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from RED_CNN_dataloader import DCMsDataset


class ST_RED_CNN(nn.Module):
    def __init__(self):
        super(ST_RED_CNN, self).__init__()
        # STN
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True))
        self.fc_loc = nn.Sequential(
            # (64x64) -conv7-> (58x58) -pool2-> (29x29) -conv5-> (25x25) -pool2-> (12x12)
            nn.Linear(10*12*12, 32),
            nn.ReLU(True),
            nn.Linear(32, 3*2))
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))
        # RED-CNN
        self.conv_first = nn.Conv2d(1, 96, kernel_size=5, stride=1, padding=0)
        self.conv = nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=0)
        self.deconv = nn.ConvTranspose2d(96, 96, kernel_size=5, stride=1, padding=0)
        self.deconv_last = nn.ConvTranspose2d(96, 1, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # stn
        stn_x = self.STN(x)
        # encoder
        residual1 = stn_x.clone()
        layer = self.relu(self.conv_first(stn_x))
        layer = self.relu(self.conv(layer))
        residual2 = layer.clone()
        layer = self.relu(self.conv(layer))
        layer = self.relu(self.conv(layer))
        residual3 = layer.clone()
        layer = self.relu(self.conv(layer))
        # decoder
        layer = self.deconv(layer)
        layer += residual3
        layer = self.deconv(self.relu(layer))
        layer = self.deconv(self.relu(layer))
        layer += residual2
        layer = self.deconv(self.relu(layer))
        layer = self.deconv_last(self.relu(layer))
        layer += residual1
        layer = self.relu(layer)
        return layer

    def STN(self, x):
        x_ = self.localization(x)
        x_ = x_.view(-1, 10*12*12)
        theta = self.fc_loc(x_)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


LEARNING_RATE = 1e-3
LEARNING_RATE_ = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 300
OUT_CHANNELS = 96

input_img_dir = '/home/datascience/PycharmProjects/CT/patch/input2/'
target_img_dir = '/home/datascience/PycharmProjects/CT/patch/target2/'

print(os.listdir(input_img_dir)[:3])
print(os.listdir(target_img_dir)[:3])

dcm = DCMsDataset(input_img_dir, target_img_dir)
dcmloader = DataLoader(dcm, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dcm, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

redcnn = ST_RED_CNN()
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), "GPUs")
    redcnn = nn.DataParallel(redcnn)

redcnn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(redcnn.parameters(), lr=LEARNING_RATE)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

total_step = len(dcmloader)
current_lr = LEARNING_RATE

# training
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
            print('Epoch [{}/{}], Step [{}/{}], Loss {:.4f}'.format(epoch+1, NUM_EPOCHS, i+1, total_step, loss.item()))

        if (epoch+1) % 10 == 0:
            current_lr /= 1.2
            update_lr(optimizer, current_lr)

torch.save(redcnn.state_dict(), 'ST_redcnn_100ep.ckpt')



