import os
import numpy as np
from torch.utils.data import Dataset


class DCMsDataset2(Dataset):
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

