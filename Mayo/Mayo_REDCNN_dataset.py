
import os
import dicom
import numpy as np
import torch

print('current directory :', os.getcwd())

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


def to_patch2D(tensor_img, patch_size, stride):
    patches = tensor_img.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    return patches




### extract patch image
my_dir = '/home/shsy0404/'
data_dir = '/data1/AAPM-Mayo-CT-Challenge/'
patch_dir = '/data1/Mayo-CT-patches'
if 'Mayo-CT-patches' not in os.listdir('/data1'):
    os.mkdir(patch_dir)

input_dir = os.path.join(patch_dir, 'input_patch/')
target_dir = os.path.join(patch_dir, 'target_patch/')
test_dir = os.path.join(patch_dir, 'test/')
if 'input_patch' not in os.listdir(patch_dir):
    os.mkdir(input_dir)
    os.mkdir(target_dir)
    os.mkdir(test_dir)

data_lst = [data for data in os.listdir(data_dir) if "L" in data and "zip" not in data]

PATCH_SIZE = 55
STRIDE = 4


for patient_ind, patient in enumerate(data_lst):
    if patient != 'L506': # input
        patient_path = os.path.join(data_dir, patient)

        patch_input_path = [data for data in os.listdir(patient_path) if "quarter" in data and "3mm" in data and "sharp" not in data][0]
        patch_target_path = [data for data in os.listdir(patient_path) if "full" in data and "3mm" in data and "sharp" not in data][0]

        for io in [patch_input_path, patch_target_path]:
            patch_pixels = get_pixels_hu(load_scan(os.path.join(patient_path,io)+'/'))
            patch_tensor = torch.Tensor(patch_pixels)
            patches = to_patch2D(patch_tensor, PATCH_SIZE, STRIDE)

            ind = 0
            for batch in range(patches.size()[0]):
                for row in range(patches.size()[1]):
                    for col in range(patches.size()[2]):
                        patch = patches[batch][row][col]
                        if io == patch_input_path:
                            np.save(input_dir + "{}_3mm_input_{}.npy".format(patient, ind), patch)
                            ind += 1
                        else:
                            np.save(target_dir + "{}_3mm_target_{}.npy".format(patient, ind), patch)
                            ind += 1
                print("{}.{} in progress.. {}/{}".format(patient_ind, patient, batch+1, patches.size()[0]))
    else: # target
        patient_path = os.path.join(data_dir, patient)

        test_input_path = [data for data in os.listdir(patient_path) if "quarter" in data and "3mm" in data and "sharp" not in data][0]
        test_target_path = [data for data in os.listdir(patient_path) if "full" in data and "3mm" in data and "sharp" not in data][0]

        for io in [test_input_path, test_target_path]:
            patch_pixels = get_pixels_hu(load_scan(os.path.join(patient_path,io)+'/'))
            for i in range(patch_pixels.shape[0]):
                if io == test_input_path:
                    np.save(test_dir + "{}_3mm_test_input_{}.npy".format(patient, i), patch_pixels[i])
                else:
                    np.save(test_dir + "{}_3mm_test_target_{}.npy".format(patient, i), patch_pixels[i])
        print("{}.{} in progress.. {}".format(patient_ind, patient, io))







### Full img
my_dir = '/home/shsy0404/'
data_dir = '/data1/AAPM-Mayo-CT-Challenge/'
full_dir = '/data1/Mayo-CT-full/'
if 'Mayo-CT-full' not in os.listdir('/data1'):
    os.mkdir(full_dir)

input_dir = os.path.join(full_dir, 'input_full_img/')
target_dir = os.path.join(full_dir, 'target_full_img/')
test_dir = '/data1/Mayo-CT-patches/test/'

data_lst = [data for data in os.listdir(data_dir) if "L" in data and "zip" not in data]

for patient_ind, patient in enumerate(data_lst):
    if patient != 'L506':
        patient_path = os.path.join(data_dir, patient)

        input_path = [data for data in os.listdir(patient_path) if "quarter" in data and "3mm" in data and "sharp" not in data][0]
        target_path = [data for data in os.listdir(patient_path) if "full" in data and "3mm" in data and "sharp" not in data][0]

        for io in [input_path, target_path]:
            full_pixels = get_pixels_hu(load_scan(os.path.join(patient_path,io)+'/'))

            if io == input_path:
                for img_ind in range(full_pixels.shape[0]):
                    np.save(input_dir + "{}_3mm_input_full_{}.npy".format(patient, img_ind), full_pixels[img_ind])
                    print("{}_input_{}/{}".format(patient, img_ind + 1, full_pixels.shape[0]))
            else:
                for img_ind in range(full_pixels.shape[0]):
                    np.save(target_dir + "{}_3mm_target_full_{}.npy".format(patient, img_ind), full_pixels[img_ind])
                    print("{}_target_{}/{}".format(patient, img_ind + 1, full_pixels.shape[0]))








### sort by input/target index
patch_file_i = os.listdir(patch_input)
patch_file_t = os.listdir(patch_target)
input_fname = [f for f in patch_file_i if f[:1] == 'i']
target_fname = [f for f in patch_file_t if f[:1] == 't']
input_fname = sorted(input_fname)
target_fname = sorted(target_fname)

print(input_fname[-10:])
print(target_fname[-10:])





############ tmp

import os
import dicom
import numpy as np
import torch

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

def to_patch2D(tensor_img, patch_size, stride):
    patches = tensor_img.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    return patches

my_dir = '/home/shsy0404/'
data_dir = '/data1/AAPM-Mayo-CT-Challenge/'
patch_dir = '/data1/Mayo-CT-patches'
input_dir = os.path.join(patch_dir, 'input_patch/')
target_dir = os.path.join(patch_dir, 'target_patch/')
data_lst = [data for data in os.listdir(data_dir) if "L" in data and "zip" not in data]

PATCH_SIZE = 55
STRIDE = 4

for patient_ind, patient in enumerate(data_lst):
    if patient == 'L067': # input
        patient_path = os.path.join(data_dir, patient)

        patch_input_path = [data for data in os.listdir(patient_path) if "quarter" in data and "3mm" in data and "sharp" not in data][0]
        patch_target_path = [data for data in os.listdir(patient_path) if "full" in data and "3mm" in data and "sharp" not in data][0]

        for io in [patch_input_path, patch_target_path]:
            patch_pixels = get_pixels_hu(load_scan(os.path.join(patient_path,io)+'/'))
            patch_tensor = torch.Tensor(patch_pixels)
            patches = to_patch2D(patch_tensor, PATCH_SIZE, STRIDE)

            ind = 0
            for batch in range(patches.size()[0]):
                for row in range(patches.size()[1]):
                    for col in range(patches.size()[2]):
                        patch = patches[batch][row][col]
                        if io == patch_input_path:
                            np.save('/data1/Mayo-CT-patches/L067_patch/input/' + "{}_3mm_input_{}.npy".format(patient, ind), patch)
                            ind += 1
                        else:
                            np.save('/data1/Mayo-CT-patches/L067_patch/target/' + "{}_3mm_target_{}.npy".format(patient, ind), patch)
                            ind += 1
                print("({}) {} {} in progress.. {}/{}".format(patient_ind, patient, io, batch+1, patches.size()[0]))

