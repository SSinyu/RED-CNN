# RED_CNN (In Progress..)
Implementation of Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)
https://arxiv.org/abs/1702.00288    

There are several things different from the original paper.
  * The input image patch(55x55 size) is extracted randomly from the 512x512 size image. --> Original : Extract patches at regular intervals from the entire image.
  * Use the loss function considering the Hounsfield (MSE loss is also used).

<img src="https://github.com/SSinyu/RED_CNN/blob/master/img/redcnn.PNG" width="550"/> 

* Comparison of LDCT and NDCT by data range. (Mean ± sd)

----|**PAPER**|-1000~400|-1000~500|-1000~600|-1000~700|-1000~800|-1000~900|-1000~1000|-1000~1100|-1000~1200
----|----|----|----|----|----|----|----|----|----|----
PSNR|**39.4314± 1.5206**|36.7279 ± 1.9180|37.2894 ± 1.9169|37.8255 ± 1.9159|38.3354 ± 1.9150|38.8200 ± 1.9143|39.2809 ± 1.1913|39.7200 ± 1.1913|40.1390 ± 1.9125|40.5395 ± 1.9119
SSIM|**0.9122 ± 0.0280**|0.8827 ± 0.0386|0.8919 ± 0.0364|0.9001 ± 0.0343|0.9075 ± 0.0323|0.9141 ± 0.0305|0.9201 ± 0.0288|0.9254 ± 0.0272|0.9303 ± 0.0257|0.9348 ± 0.0243
RMSE|**0.0109 ± 0.0021**|0.0149 ± 0.0032|0.0139 ± 0.0030|0.0131 ± 0.0028|0.0124 ± 0.0027|0.0117 ± 0.0025|0.0111 ± 0.0024|0.0105 ± 0.0023|0.0100 ± 0.0021|0.0096 ± 0.0020


* PSNR, SSIM, RMSE value on 1 epoch
  - blue: 55x55 patch image learning
  - yellow: 512x512 full image learning

<img src="https://github.com/SSinyu/RED_CNN/blob/master/img/PSNR_comp.PNG" width="250"/> <img src="https://github.com/SSinyu/RED_CNN/blob/master/img/SSIM_comp.PNG" width="250"/> <img src="https://github.com/SSinyu/RED_CNN/blob/master/img/RMSE_comp.PNG" width="250"/> 
  
* (paper) 55x55 img input  

----|**PAPER LDCT**|**PAPER OUTPUT**
----|----|----
PSNR|**39.4314 ± 1.5206**|**44.4187 ± 1.2118**
SSIM|**0.9122 ± 0.0280**|**0.9705 ± 0.0087**
RMSE|**0.0109 ± 0.0021**|**0.0060 ± 0.0009**

* 55x55 patch input

----|**LDCT**|100ep|200ep|300ep|400ep|500ep|
----|----|----|----|----|----|----
PSNR|**47.1245 ± 1.7507**|51.2305 ± 1.3865|51.3291 ± 1.4168|**51.3323 ± 1.4017**|51.3210 ± 1.3943|51.3253 ± 1.3950|
SSIM|**0.9809 ± 0.0093**|0.9928 ± 0.0029|**0.9930 ± 0.0029**|0.9930 ± 0.0029|0.9929 ± 0.0029|0.9930 ± 0.0029|
RMSE|**18.4390 ± 4.2333**|11.3958 ± 2.0135|11.2741 ± 2.0382|**11.2663 ± 2.0115**|11.2792 ± 2.0020|11.2738 ± 2.0024|



* 55x55 Hounsfield loss

----|**LDCT**|100ep|200ep|300ep|400ep|500ep|
----|----|----|----|----|----|----
PSNR|**47.1245 ± 1.7507**|51.1134 ± 1.3683|51.2038 ± 1.3726|**51.2712 ± 1.3942**|51.2482 ± 1.3806|51.2566 ± 1.3846|
SSIM|**0.9809 ± 0.0093**|0.9927 ± 0.0029|0.9928 ± 0.0029|**0.9929 ± 0.0029**|0.9928 ± 0.0029|0.9929 ± 0.0029|
RMSE|**18.4390 ± 4.2333**|11.5462 ± 2.0113|11.4277 ± 1.9987|**11.3442 ± 2.0142**|11.3712 ± 1.9974|11.3610 ± 2.0018|




* Visualization

<img src="https://github.com/SSinyu/RED_CNN/blob/master/img/vis.png">

