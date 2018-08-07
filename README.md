# RED_CNN
Implementation of Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)
https://arxiv.org/abs/1702.00288    

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
PSNR|**40.2252**|**45.2616**
SSIM|**0.9292**|**0.9764**
RMSE|**0.0097**|**0.0055**

* 55x55 img input  

----|**LDCT**|100ep|200ep|300ep|400ep|500ep
----|----|----|----|----|----|----
PSNR|**47.1145**|51.0987|51.0645|51.0225|51.1208|**51.2087**
SSIM|**0.9812**|0.9893|0.9893|0.9892|0.9893|**0.9894**
RMSE|**18.4802**|15.2714|15.3068|15.3602|15.2387|**15.1332**

* 512x512 img input  

----|**LDCT**|100ep|200ep|300ep|400ep|500ep
----|----|----|----|----|----|----
PSNR|**47.1145**|51.0699|51.1611|51.1173|**51.1671**|51.1668
SSIM|**0.9812**|0.9893|0.9894|0.9893|0.9893|**0.9894**
RMSE|**18.4802**|15.3025|15.1783|15.1788|15.1698|**15.1535**
  
 
![ct3](https://github.com/SSinyu/RED_CNN/blob/master/img/mayo-full.PNG)
