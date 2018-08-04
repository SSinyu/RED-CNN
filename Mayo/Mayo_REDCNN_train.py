import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Mayo_REDCNN_dataloader import DCMsDataset2
from Mayo_REDCNN_model import RED_CNN

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def main():
    ### full img training
    full_data_path = '/home/shsy0404/Mayo-CT-full/'
    full_data_dir = sorted(os.listdir(full_data_path))
    input_img_dir = full_data_path + full_data_dir[1]
    target_img_dir = full_data_path + full_data_dir[3]

    input_dir = []
    for f in sorted(os.listdir(input_img_dir)):
        input_dir.append(np.load(input_img_dir + '/' + f))
    target_dir = []
    for f in sorted(os.listdir(target_img_dir)):
        target_dir.append(np.load(target_img_dir + '/' + f))

    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 1000
    BATCH_SIZE = 10
    OUT_CHANNELS = 96

    dcm = DCMsDataset2(input_dir, target_dir, crop_size=55, crop_n=10)
    dcm_loader = DataLoader(dcm, batch_size=BATCH_SIZE, shuffle=True, num_workers=30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    red_cnn = RED_CNN(OUT_CHANNELS)

    if torch.cuda.device_count() > 1:
        print("Use {} GPUs".format(torch.cuda.device_count()), "=" * 10)
        red_cnn = nn.DataParallel(red_cnn)
    red_cnn.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(red_cnn.parameters(), lr=LEARNING_RATE)

    total_step = len(dcm_loader)
    current_lr = LEARNING_RATE

    for epoch in range(NUM_EPOCHS):
        for i, (inputs, targets) in enumerate(dcm_loader):
            inputs = inputs.reshape(-1, 55, 55).to(device)
            targets = targets.reshape(-1, 55, 55).to(device)
            input_img = torch.tensor(inputs, requires_grad=True, dtype=torch.float32).unsqueeze(1).to(device)
            target_img = torch.tensor(targets, dtype=torch.float32).unsqueeze(1).to(device)

            outputs = red_cnn(input_img)
            loss = criterion(outputs, target_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('EPOCH [{}/{}], STEP [{}/{}], LOSS {:.4f}'.format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

            if (epoch + 1) % 100 == 0:
                torch.save(red_cnn.state_dict(), '/home/shsy0404/red_cnn_mayo_patch_{}ep.ckpt'.format(epoch + 1))

                current_lr /= 1.1
                update_lr(optimizer, current_lr)


if __name__ == "__main__":
    main()

