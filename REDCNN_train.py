import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from RED_CNN_util import build_dataset, RED_CNN, train_dcm_data_loader
from RED_CNN_loss import H_loss2


#os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2, 3"
#save_path = '/home/shsy0404/result/REDCNN_result/'

LEARNING_RATE = 1e-4
NUM_EPOCHS = 500
DECAY_EPOCHS = 100
BATCH_SIZE = 3
CROP_NUMBER = 60  # The number of patches to extract from a single image. --> total batch img is BATCH_SIZE * CROP_NUMBER
CROP_SIZE = 55
N_CPU = 30
d_min = -1024.0
d_max = 3072.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    input_dir, target_dir, test_input_dir, test_target_dir = build_dataset(['L067','L291'], "3mm", norm_range=(-1024.0, 3072.0))
    train_dcm = train_dcm_data_loader(input_dir, target_dir, crop_size=CROP_SIZE, crop_n=CROP_NUMBER)
    train_loader = DataLoader(train_dcm, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CPU, drop_last=True)

    # multi gpu
    red_cnn = RED_CNN()
    if torch.cuda.device_count() > 1:
        print("Use {} GPUs".format(torch.cuda.device_count()), "=" * 10)
        red_cnn = nn.DataParallel(red_cnn)
    red_cnn.to(device)

    # define loss function and optimizer
    criterion = nn.MSELoss()
    criterion_H = H_loss2()
    optimizer = torch.optim.Adam(red_cnn.parameters(), lr=LEARNING_RATE)

    # save loss
    loss_ = []

    # training
    for epoch in range(NUM_EPOCHS):

        loss_lst = train(train_loader, red_cnn, criterion, optimizer, epoch)
        loss_.extend(loss_lst)

        if (epoch + 1) % 10 == 0:
            if CROP_NUMBER == None:
                torch.save(red_cnn.state_dict(), os.path.join(save_path, 'redcnn_full_{}ep.ckpt'.format(epoch)))
            else:
                torch.save(red_cnn.state_dict(), os.path.join(save_path, 'redcnn_patch_{}ep.ckpt'.format(epoch)))

            result_ = {'Loss': loss_}
            with open(os.path.join(save_path, '{}_losslist.pkl'.format(epoch)), 'wb') as f:
                pickle.dump(result_, f)

        if (epoch) % DECAY_EPOCHS == 0:
            adjust_learning_rate(optimizer, epoch, DECAY_EPOCHS)



def train(data_loader, model, criterion, optimizer, epoch, CROP_tf=True):
    model.train()
    loss_lst = []
    for i, (inputs, targets) in enumerate(data_loader):
        # patch training?
        if CROP_tf == True:
            inputs = inputs.reshape(-1, CROP_SIZE, CROP_SIZE).to(device)
            targets = targets.reshape(-1, CROP_SIZE, CROP_SIZE).to(device)
        input_img = torch.tensor(inputs, requires_grad=True, dtype=torch.float32).unsqueeze(1).to(device)
        target_img = torch.tensor(targets, dtype=torch.float32).unsqueeze(1).to(device)

        outputs = model(input_img)
        loss = criterion(outputs, target_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # every iteration loss save
        loss_lst.append(loss.item())

        if (i + 1) % 10 == 0:
            print('EPOCH [{}/{}], STEP [{}/{}], LOSS {:.4f}'.format(epoch + 1, NUM_EPOCHS, i + 1, len(data_loader), loss.item()*10000))

    return loss_lst

def adjust_learning_rate(optimizer, epoch, DECAY_EPOCH):
    lr = LEARNING_RATE * (0.1 ** (epoch // DECAY_EPOCH))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == "__main__":
    main()
