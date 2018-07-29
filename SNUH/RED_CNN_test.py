
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from RED_CNN_model import RED_CNN
from skimage.measure import compare_psnr, compare_ssim, compare_nrmse

def test_RED_CNN(data_path, num_test, pre_model='redcnn_30ep.ckpt', model=RED_CNN(), figure=None, MultiGPU=None):
    # model load
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    redcnn = model.to(device)
    if MultiGPU:
        from collections import OrderedDict
        state_dict = torch.load(pre_model)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        redcnn.load_state_dict(new_state_dict)
    else:
        redcnn.load_state_dict(torch.load(pre_model))

    # data load
    dir = sorted(os.listdir(data_path))
    test_inputs = []; test_targets = []
    for fname in dir:
        if 'input' in fname:
            test_inputs.append(fname)
        else: test_targets.append(fname)

    random_test = sorted(np.random.randint(len(test_inputs), size=num_test))

    test_img_ori = [np.load(data_path+sorted(test_inputs)[i]) for i in random_test]
    test_img = [torch.Tensor(img).unsqueeze(0).unsqueeze(0) for img in test_img_ori]

    #test_output = []
    #for img in test_img:
    #    red_output = redcnn(img.to(device))
    #    torch.cuda.empty_cache()
    #    test_output.append(red_output)
    test_output =[redcnn(img.to(device)) for img in test_img]
    test_output = [img.squeeze(0).squeeze(0) for img in test_output]
    test_output_img = [img.data.cpu().numpy() for img in test_output]

    test_target = [np.load(data_path+sorted(test_targets)[i]) for i in random_test]

    if figure != None:
        fig = plt.figure(figsize=(40,40))
        i = 0
        for img1, img2, img3 in zip(test_img_ori, test_output_img, test_target):
            for img_ in [img1, img2, img3]:
                fig.add_subplot(len(random_test), 3, i+1)
                plt.imshow(img_, cmap=plt.cm.gray)
                i += 1
    return test_img_ori, test_output_img, test_target


test_path = '/home/datascience/PycharmProjects/CT/dev_image/'
pre_model1 = 'redcnn_30ep.ckpt'
pre_model2 = 'redcnn_100ep.ckpt'

ori, test, tar = test_RED_CNN(test_path, 2, pre_model=pre_model1)
ori2, test2, tar2 = test_RED_CNN(test_path, 2, pre_model=pre_model2)

compare_psnr(ori3, tar3)
compare_psnr(test3, tar3)

compare_ssim(ori3[0], tar3[0])
compare_nrmse(np.array(ori2), np.array(tar2))
compare_nrmse(np.array(test2), np.array(tar2))



'''
### TEST
# model load
device = torch.device('cuda')
redcnn_test = RED_CNN().to(device)
redcnn_test.load_state_dict(torch.load('redcnn_30ep.ckpt'))
# data load
test_path = '/home/datascience/PycharmProjects/CT/dev_image/'
test_dir = sorted(os.listdir(test_path))
test_input = []; test_targets = []
for file_name in test_dir:
    if 'input' in file_name:
        test_input.append(file_name)
    else: test_targets.append(file_name)
#test_img_ori = cv2.imread(np.load(test_path+sorted(test_input)[80]), 0)
test_img_ori = [np.load(test_path+sorted(test_input)[200]),
                np.load(test_path+sorted(test_input)[180]),
                np.load(test_path+sorted(test_input)[130])]
print(test_img_ori[0].shape)
test_img = [torch.Tensor(img).unsqueeze(0).unsqueeze(0) for img in test_img_ori]
print(test_img[0].size())
test_output = [redcnn_test(img.to(device)) for img in test_img]
test_output = [img.squeeze(0).squeeze(0) for img in test_output]
test_output[0].size()
test_output_img = [img.data.cpu().numpy() for img in test_output]
type(test_output_img[0])
#test_target = cv2.imread(test_path+sorted(test_target)[80], 0)
test_target = [np.load(test_path+sorted(test_targets)[200]),
               np.load(test_path+sorted(test_targets)[180]),
               np.load(test_path+sorted(test_targets)[130])]
print(test_target[0].shape)
fig = plt.figure(figsize=(40,40))
i = 0
for img1, img2, img3 in zip(test_img_ori, test_output_img, test_target):
    fig.add_subplot(3,3,i+1)
    plt.imshow(img1, cmap=plt.cm.gray)
    fig.add_subplot(3,3,i+2)
    plt.imshow(img2, cmap=plt.cm.gray)
    fig.add_subplot(3,3,i+3)
    plt.imshow(img3, cmap=plt.cm.gray)
    i += 3
'''


'''
def psnr(input_img, output_img, pixel_max=4095):
    # require to numpy array
    mse = np.mean((input_img - output_img) ** 2)
    if mse == 0:
        return 100
    return 20*math.log10(pixel_max/math.sqrt(mse))
'''
