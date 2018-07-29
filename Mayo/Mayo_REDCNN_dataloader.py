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


class Multi_dir_DCMsDataset(Dataset):
    def __init__(self, input_path_lst, target_path_lst, transform=None):
        self.input_path_lst = input_path_lst
        self.target_path_lst = target_path_lst
        self.input_dir = sorted([i_path + '/' + img for i_path in input_path_lst for img in os.listdir(i_path)])
        self.target_dir = sorted([i_path + '/' + img for i_path in target_path_lst for img in os.listdir(i_path)])
        self.transform = transform

    def __len__(self):
        return len(self.input_dir)

    def __getitem__(self, idx):
        input_name = self.input_dir[idx]
        target_name = self.target_dir[idx]
        input_img = np.load(input_name)
        target_img = np.load(target_name)
        #sample = {'input':input_img, 'target':target_img}
        sample = (input_img, target_img)

        if self.transform:
            sample = self.transform(sample)

        return sample



class Multi_dir_(Dataset):
    def __init__(self, input_path_lst, target_path_lst):
        self.input_dir = input_path_lst
        self.target_dir = target_path_lst

    def __len__(self):
        return len(self.input_dir)

    def __getitem__(self, idx):
        input_name = self.input_dir[idx]
        target_name = self.target_dir[idx]
        input_img = np.load(input_name)
        target_img = np.load(target_name)
        sample = (input_img, target_img)
        return sample



class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

