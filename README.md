# RED_CNN
Implementation of Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)  

<img src="https://github.com/SSinyu/RED_CNN/blob/master/img/redcnn.PNG" width="550"/> 

There is several things different from the original paper.
  * The input image patch(64x64 size) is extracted randomly from the 512x512 size image. --> Original : Extract patches at regular intervals from the entire image.
  * use Adam optimizer
  
### DATASET

The 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge by Mayo Clinic (I can't share this data, you should ask at the URL below if you want)  
https://www.aapm.org/GrandChallenge/LowDoseCT/

The data_path should look like:


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

### RESULT  

<img src="https://github.com/SSinyu/RED_CNN/blob/master/img/result_12.png">
<img src="https://github.com/SSinyu/RED_CNN/blob/master/img/result_29.png">
<img src="https://github.com/SSinyu/RED_CNN/blob/master/img/result_47.png">

