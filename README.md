# RED_CNN
Implementation of [Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)](https://arxiv.org/ftp/arxiv/papers/1702/1702.00288.pdf)  

<img src="https://github.com/SSinyu/RED_CNN/blob/master/img/redcnn.PNG" width="550"/> 

There is several things different from the original paper.
  * The input image patch(64x64 size) is extracted randomly from the 512x512 size image. --> Original : Extract patches at regular intervals from the entire image.
  * use Adam optimizer

-----

### DATASET

The 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge by Mayo Clinic   
(I can't share this data, you should ask at the URL below if you want)  
https://www.aapm.org/GrandChallenge/LowDoseCT/

The `data_path` should look like:


    data_path
    ├── L067
    │   ├── quarter_3mm
    │   │       ├── L067_QD_3_1.CT.0004.0001 ~ .IMA
    │   │       ├── L067_QD_3_1.CT.0004.0002 ~ .IMA
    │   │       └── ...
    │   └── full_3mm
    │           ├── L067_FD_3_1.CT.0004.0001 ~ .IMA
    │           ├── L067_FD_3_1.CT.0004.0002 ~ .IMA
    │           └── ...
    ├── L096
    │   ├── quarter_3mm
    │   │       └── ...
    │   └── full_3mm
    │           └── ...      
    ...
    │
    └── L506
        ├── quarter_3mm
        │       └── ...
        └── full_3mm
                └── ...     

-------

## Use
Check the arguments.

1. run `python prep.py` to convert 'dicom file' to 'numpy array'
2. run `python main.py --load_mode=0` to training. If the available memory(RAM) is more than 10GB, it is faster to run `--load_mode=1`.
3. run `python main.py --mode='test' --test_iters=100000` to test.


-------

### RESULT  

<img src="https://github.com/SSinyu/RED_CNN/blob/master/img/result_11.png" width="800"/>   
<img src="https://github.com/SSinyu/RED_CNN/blob/master/img/result_25.png" width="800"/>   
<img src="https://github.com/SSinyu/RED_CNN/blob/master/img/result_81.png" width="800"/>   