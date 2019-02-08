# RED_CNN
Implementation of Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)  

<img src="https://github.com/SSinyu/RED_CNN/blob/master/img/redcnn.PNG" width="550"/> 

There is several things different from the original paper.
  * The input image patch(55x55 size) is extracted randomly from the 512x512 size image. --> Original : Extract patches at regular intervals from the entire image.
  * Masking the Hounsfield unit corresponding to the air and calculating the MSE loss. (The original MSE loss also proceeds)
  * 512x512 entire image input is used without extracting the patches.

### DATASET
“the 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge” by Mayo Clinic  
https://www.aapm.org/GrandChallenge/LowDoseCT/

The data_path directory should look like:


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

<img src="https://github.com/SSinyu/RED_CNN/blob/master/img/REDCNN_full_result.png">
<img src="https://github.com/SSinyu/RED_CNN/blob/master/img/REDCNN_ROI_result.png">

