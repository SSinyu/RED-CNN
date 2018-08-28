import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Mayo_REDCNN_dataloader import train_dcm_data_loader, validate_dcm_data_loader
from Mayo_REDCNN_model import RED_CNN
from skimage.measure import compare_mse, compare_psnr, compare_ssim

LEARNING_RATE = 1e-4
NUM_EPOCHS = 1000
BATCH_SIZE = 5
CROP_NUMBER = 40 # The number of patches to extract from a single image. --> total batch size is BATCH_SIZE * CROP_NUMBER
PATCH_SIZE = 55
NUM_WORKERS = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # train data (2167 image)
    full_data_path = '/home/shsy0404/Mayo-CT-full/'
    input_img_dir = full_data_path + 'input_full_img'
    target_img_dir = full_data_path + 'target_full_img'

    input_dir = []; target_dir = []
    for f in sorted(os.listdir(input_img_dir)):
        input_dir.append(np.load(input_img_dir + '/' + f))
    for f in sorted(os.listdir(target_img_dir)):
        target_dir.append(np.load(target_img_dir + '/' + f))

    # validation data (11 image)
    val_data_path = '/home/shsy0404/Mayo-CT-full/test/'
    input_val_dir_ = sorted([data for data in os.listdir(val_data_path) if 'input' in data])
    target_val_dir_ = sorted(list(set(os.listdir(val_data_path)) - set(input_val_dir_)))

    input_val_dir = []; target_val_dir = []
    for (inp, tar) in zip(input_val_dir_[-11:], target_val_dir_[-11:]):
        inp_ = np.load(val_data_path + inp)
        inp_ = np.float32(inp_)
        tar_ = np.load(val_data_path + tar)
        tar_ = np.float32(tar_)
        input_val_dir.append(inp_)
        target_val_dir.append(tar_)

    # train, validate data loading
    train_dcm = train_dcm_data_loader(input_dir, target_dir, crop_size=PATCH_SIZE, crop_n=CROP_NUMBER)
    train_loader = DataLoader(train_dcm, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_dcm = validate_dcm_data_loader(input_val_dir, target_val_dir)
    val_loader = DataLoader(val_dcm, batch_size=1, num_workers=NUM_WORKERS)

    # multi gpu
    red_cnn = RED_CNN()
    if torch.cuda.device_count() > 1:
        print("Use {} GPUs".format(torch.cuda.device_count()), "=" * 10)
        red_cnn = nn.DataParallel(red_cnn)
    red_cnn.to(device)

    # define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(red_cnn.parameters(), lr=LEARNING_RATE)

    # save values
    loss_ = []; rmse = []; psnr = []; ssim = []

    # training
    for epoch in range(NUM_EPOCHS):

        loss_lst = train(train_loader, red_cnn, criterion, optimizer, epoch)
        loss_.extend(loss_lst)

        r, p, s = validate(val_loader, red_cnn)
        rmse.append(r)
        psnr.append(p)
        ssim.append(s)

        if (epoch + 1) % 100 == 0:
            if CROP_NUMBER == None:
                torch.save(red_cnn.state_dict(), '/home/shsy0404/red_cnn_mayo_{}ep.ckpt'.format(epoch + 1))
            else:
                torch.save(red_cnn.state_dict(), '/home/shsy0404/red_cnn_mayo_patch_{}ep.ckpt'.format(epoch + 1))

    result_ = {'Loss':loss_, 'RMSE':rmse, 'PSNR':psnr, 'SSIM':ssim}
    with open('/home/shsy0404/result_.pkl', 'wb') as f:
        pickle.dump(result_, f)



def train(data_loader, model, criterion, optimizer, epoch, CROP_tf=None):
    model.train()
    loss_lst = []
    for i, (inputs, targets) in enumerate(data_loader):
        if CROP_tf != None:
            inputs = inputs.reshape(-1, PATCH_SIZE, PATCH_SIZE).to(device)
            targets = targets.reshape(-1, PATCH_SIZE, PATCH_SIZE).to(device)
        input_img = torch.tensor(inputs, requires_grad=True, dtype=torch.float32).unsqueeze(1).to(device)
        target_img = torch.tensor(targets, dtype=torch.float32).unsqueeze(1).to(device)

        outputs = model(input_img)
        loss = criterion(outputs, target_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_lst.append(loss.item())

        if (i + 1) % 10 == 0:
            print('EPOCH [{}/{}], STEP [{}/{}], LOSS {:.4f}'.format(epoch + 1, NUM_EPOCHS, i + 1, len(data_loader), loss.item()))

    return loss_lst




def validate(data_loader, model):
    model.eval()

    with torch.no_grad():
        rmse_lst = []; psnr_lst = []; ssim_lst = []

        for i, (inputs, targets) in enumerate(data_loader):
            input_img = inputs.to(device).unsqueeze(0)
            target_img = targets.to(device)

            output_img = model(input_img)
            output_img = output_img.squeeze(0).squeeze(0)
            output_img = output_img.data.cpu().numpy()

            target_img = target_img.squeeze(0)
            target_img = target_img.data.cpu().numpy()

            rmse_lst.append(np.sqrt(compare_mse(output_img, target_img)))
            psnr_lst.append(compare_psnr(output_img, target_img, data_range=4096))
            ssim_lst.append(compare_ssim(output_img, target_img, data_range=4096))

        rmse_avg = np.mean(rmse_lst)
        psnr_avg = np.mean(psnr_lst)
        ssim_avg = np.mean(ssim_lst)
        print('RMSE [{:.4f}], PSNR [{:.4f}], SSIM [{:.4f}]'.format(
            rmse_avg, psnr_avg, ssim_avg))

    return rmse_avg, psnr_avg, ssim_avg



def adjust_learning_rate(optimizer, epoch):
    lr = LEARNING_RATE * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == "__main__":
    main()

