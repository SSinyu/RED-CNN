import torch.nn as nn

class H_loss1(nn.Module):
    def __init__(self):
        super(H_loss1, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, input_img, target_img):
        img_mean = target_img.mean()
        return self.criterion(input_img, target_img) * img_mean


class H_loss2(nn.Module):
    def __init__(self, threshold):
        super(H_loss2, self).__init__()
        self.criterion = nn.MSELoss()
        self.threshold = threshold

    def forward(self, input_img, target_img):
        input_img[input_img < self.threshold] = 0
        target_img[target_img < self.threshold] = 0
        return self.criterion(input_img, target_img)
