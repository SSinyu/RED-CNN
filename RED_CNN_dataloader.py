import os
import torch
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


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



# build dir for tmptestset copy
#test_path = '/home/datascience/PycharmProjects/CT/test/'
#os.mkdir(test_path)
#for i in range(100):
#	img = np.load(patch_path+patch_list[i])
#	np.save(test_path+'{}img.npy'.format(i), img)

'''
## patch 
patch_path = '/home/datascience/PycharmProjects/CT/patch/'
patch_list = os.listdir(patch_path+'input/')
patch_list[0]
test_ct = np.load(patch_path + 'input/' + patch_list[0])
print(test_ct.shape)
plt.imshow(test_ct, cmap=plt.cm.gray)

## test
dcm = DCMsDataset(root_dir=patch_path)
fig = plt.figure()
for i in range(len(dcm)):
	img = dcm[i]
	print(i, img['image'].shape)
	ax = plt.subplot(1,4,i+1)
	plt.imshow(img['image'], cmap=plt.cm.gray)
	ax.set_title('Img #{}'.format(i))
	if i==3:
		plt.show()
		break

fig = plt.figure()
for i in range(4):
	#print(patch_path+patch_list[i])
	ax = plt.subplot(1,4,i+1)
	img = np.load(patch_path+patch_list[i])
	plt.imshow(img, cmap=plt.cm.gray)
	ax.set_title(patch_list[i])


# iterating test through the dataset
patch_path = '/home/datascience/PycharmProjects/CT/patch/'
dcm = DCMsDataset(patch_path+'input/', patch_path+'target/')
dcmloader = DataLoader(dcm, batch_size=20, shuffle=False)
for i_batch, sample_batch in enumerate(dcmloader):
    print(i_batch, sample_batch['input'].size(), sample_batch['target'].size())

'''


