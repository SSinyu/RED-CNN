import os
import numpy as np
from torch.utils.data import Dataset


class train_dcm_data_loader(Dataset):
    def __init__(self, input_lst, target_lst, crop_size=None, crop_n=None):
        self.input_lst = input_lst
        self.target_lst = target_lst
        self.crop_size = crop_size
        self.crop_n = crop_n

    def __getitem__(self, idx):
        input_img = self.input_lst[idx]
        target_img = self.target_lst[idx]

        if self.crop_n:
            assert input_img.shape == target_img.shape
            crop_input = []
            crop_target = []
            h, w = input_img.shape
            new_h, new_w = self.crop_size, self.crop_size
            for _ in range(self.crop_n):
                top = np.random.randint(0, h-new_h)
                left = np.random.randint(0, w-new_w)
                input_img_ = input_img[top:top+new_h, left:left+new_w]
                target_img_ = target_img[top:top+new_h, left:left+new_w]
                crop_input.append(input_img_)
                crop_target.append(target_img_)
            crop_input = np.array(crop_input)
            crop_target = np.array(crop_target)

            sample = (crop_input, crop_target)
            return sample
        else:
            sample = (input_img, target_img)
            return sample

    def __len__(self):
        return len(self.input_lst)


class validate_dcm_data_loader(Dataset):
    def __init__(self, input_lst, target_lst):
        self.input_lst = input_lst
        self.target_lst = target_lst

    def __getitem__(self, idx):
        input_img = self.input_lst[idx]
        #input_img = torch.Tensor(input_img).unsqueeze(0)
        target_img = self.target_lst[idx]

        sample = (input_img, target_img)
        return sample

    def __len__(self):
        return len(self.input_lst)


