# RED_CNN
Implementation of Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)
https://arxiv.org/abs/1702.00288  


![PSNR](https://github.com/SSinyu/RED_CNN/blob/master/img/PSNR_comp.PNG)
![SSIM](https://github.com/SSinyu/RED_CNN/blob/master/img/SSIM_comp.PNG)
![RMSE](https://github.com/SSinyu/RED_CNN/blob/master/img/RMSE_comp.PNG)
  
  
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
