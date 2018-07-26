import os
import numpy as np
from torch.utils.data import Dataset


class DCMsDataset(Dataset):
    def __init__(self, input_path, target_path, transfrom=None):
        self.input_path = input_path
        self.target_path = target_path
        self.input_dir = sorted(os.listdir(input_path))
        self.target_dir = sorted(os.listdir(target_path))
        self.transfrom = transfrom

    def __len__(self):
        # input and target have same length
        return len(self.input_dir)

    def __getitem__(self, idx):
        input_name = os.path.join(self.input_path, self.input_dir[idx])
        target_name = os.path.join(self.target_path, self.target_dir[idx])
        input_img = np.load(input_name)
        target_img = np.load(target_name)
        #sample = {'input':input_img, 'target':target_img}
        sample = (input_img, target_img)

        if self.transfrom:
            sample = self.transfrom(sample)

        return sample
