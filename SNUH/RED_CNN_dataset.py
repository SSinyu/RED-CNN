
import os
import dicom
import numpy as np
import torch

def load_scan(path):
    slices = [dicom.read_file(path + s) for s in os.listdir(path)]
    slices.sort(key=lambda x:float(x.ImagePositionPatient[2]))
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


### extract patch image from input/target data, and save
input_path = '/home/datascience/Denoising/DENOISING MODEL_DICOM file/100KV 1ST/10 B STANDARD iDose/'
target_path = '/home/datascience/Denoising/DENOISING MODEL_DICOM file/100KV 1ST/200 B STANDARD iDose/'
patch_input = '/home/datascience/PycharmProjects/CT/patch/input/'
patch_target = '/home/datascience/PycharmProjects/CT/patch/target/'

PATCH_SIZE = 64
STRIDE = 20

def to_patch2D(tensor_img, patch_size, stride):
    patches = tensor_img.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    return patches

for io in [input_path, target_path]:
    patch_pixels = get_pixels_hu(load_scan(io))
    patch_tensor = torch.Tensor(patch_pixels)
    patches = to_patch2D(patch_tensor, PATCH_SIZE, STRIDE)
    #print(patches.size())
    ind = 0
    for batch in range(patches.size()[0]):
        for row in range(patches.size()[1]):
            for col in range(patches.size()[2]):
                patch = patches[batch][row][col]
                if io == input_path:
                    np.save(patch_input + 'input_{}.npy'.format(ind), patch)
                    ind += 1
                else:
                    np.save(patch_target + 'target_{}.npy'.format(ind), patch)
                    ind += 1
        print('{}/{} batch'.format(batch+1, patches.size()[0]))
    print('Tensor size :', patches.size())

    
### sort by input/target index
patch_file_i = os.listdir(patch_input)
patch_file_t = os.listdir(patch_target)
input_fname = [f for f in patch_file_i if f[:1] == 'i']
target_fname = [f for f in patch_file_t if f[:1] == 't']
input_fname = sorted(input_fname)
target_fname = sorted(target_fname)

print(input_fname[-10:])
print(target_fname[-10:])


### test data generate for model evaluation
test_input_path = '/home/datascience/Denoising/DENOISING MODEL_DICOM file/100KV 1ST/10 Y SHARP iDose/'
test_target_path = '/home/datascience/Denoising/DENOISING MODEL_DICOM file/100KV 1ST/200 Y SHARP iDose/'
output_path = '/home/datascience/PycharmProjects/CT/dev_image/'

for io in [test_input_path, test_target_path]:
    patch_pixels = get_pixels_hu(load_scan(io))
    for i in range(patch_pixels.shape[0]):
        if io == test_input_path:
            np.save(output_path+'raw_input_{}.npy'.format(i), patch_pixels[i])
        else:
            np.save(output_path+'raw_target_{}.npy'.format(i), patch_pixels[i])
