import os
import dicom
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

def build_dataset(test_patient_number_list, mm, norm_range=(-1024.0, 3072.0)):
    data_path = '/data1/AAPM-Mayo-CT-Challenge/'
    patients_list = [data for data in os.listdir(data_path) if 'zip' not in data]
    assert len(patients_list) == 10

    input_img = []
    target_img = []
    test_input_img = []
    test_target_img = []

    for patient_ind, patient in enumerate(patients_list):
        patient_path = os.path.join(data_path, patient)

        if patient not in test_patient_number_list:
            input_path = [data for data in os.listdir(patient_path) if "quarter" in data and mm in data and "sharp" not in data][0]
            target_path = [data for data in os.listdir(patient_path) if "full" in data and mm in data and "sharp" not in data][0]

            for io in [input_path, target_path]:
                full_pixels = get_pixels_hu(load_scan(os.path.join(patient_path, io) + '/'))
                if io == input_path:
                    for img_ind in range(full_pixels.shape[0]):
                        input_img.append(NORMalize(full_pixels[img_ind], norm_range[0], norm_range[1]))
                else:
                    for img_ind in range(full_pixels.shape[0]):
                        target_img.append(NORMalize(full_pixels[img_ind], norm_range[0], norm_range[1]))
        else:
            test_input_path = [data for data in os.listdir(patient_path) if "quarter" in data and mm in data and "sharp" not in data][0]
            test_target_path  = [data for data in os.listdir(patient_path) if "full" in data and mm in data and "sharp" not in data][0]
            for io in [test_input_path, test_target_path]:
                full_pixels = get_pixels_hu(load_scan(os.path.join(patient_path, io) + '/'))
                if io == test_input_path:
                    for img_ind in range(full_pixels.shape[0]):
                        test_input_img.append(NORMalize(full_pixels[img_ind], norm_range[0], norm_range[1]))
                else:
                    for img_ind in range(full_pixels.shape[0]):
                        test_target_img.append(NORMalize(full_pixels[img_ind], norm_range[0], norm_range[1]))

    return input_img, target_img, test_input_img, test_target_img


def load_scan(path):
    slices = [dicom.read_file(path + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def NORMalize(image, MIN_B, MAX_B):
   image = (image - MIN_B) / (MAX_B - MIN_B)
   #image[image>1] = 1.
   #image[image<0] = 0.
   return image

def DENORMalize(image, MIN_B, MAX_B):
    image = image * (MAX_B - MIN_B) + MIN_B
    return image



class train_dcm_data_loader(Dataset):
    def __init__(self, input_lst, target_lst, crop_n=None, crop_size=55):
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



class RED_CNN(nn.Module):
    def __init__(self, out_channels=96):
        super(RED_CNN, self).__init__()
        self.conv_first = nn.Conv2d(1, out_channels, kernel_size=5, stride=1, padding=0)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=0)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=5, stride=1, padding=0)
        self.deconv_last = nn.ConvTranspose2d(out_channels, 1, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        residual1 = x.clone()
        layer = self.relu(self.conv_first(x))
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
