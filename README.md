# RED_CNN (In Progress..)
Implementation of Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)
https://arxiv.org/abs/1702.00288    

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

----|**LDCT**|100ep|200ep|300ep|400ep|500ep|600ep|700ep|800ep|900ep|1000ep
----|----|----|----|----|----|----|----|----|----|----|----
PSNR|**47.1245 ± 1.7507**|51.2170 ± 1.3671|51.2584 ± 1.3765|51.3605 ± 1.4205|51.3702 ± 1.4060|51.4002 ± 1.4165|51.3988 ± 1.4247|51.4015 ± 1.4193|**51.4130 ± 1.4243**|51.3287 ± 1.3877|51.3434 ± 1.3883 
SSIM|**0.9809 ± 0.0093**|0.9928 ± 0.0029|0.9928 ± 0.0028|0.9929 ± 0.0029|0.9930 ± 0.0029|**0.9931 ± 0.0029**|0.9930 ± 0.0029|0.9930 ± 0.0029|0.9930 ± 0.0029|0.9929 ± 0.0028|0.9930 ± 0.0028
RMSE|**18.4390 ± 4.2333**|11.4088 ± 1.9798|11.3567 ± 1.9857|11.2341 ± 2.0342|11.2182 ± 2.0094|11.1818 ± 2.0170|11.1856 ± 2.0314|**11.1809 ± 2.0237**|11.1672 ± 2.0277|11.2685 ± 1.9886|11.2487 ± 1.9842
SSIM(Norm)|**0.9814 ± 0.0093**|0.9933 ± 0.0029|0.9934 ± 0.0027|0.9934 ± 0.0027|0.9931 ± 0.0029|**0.9936 ± 0.0027**|0.9936 ± 0.0027|0.9936 ± 0.0027|0.9936 ± 0.0027|0.9936 ± 0.0026|0.9936 ± 0.0.0026
RMSE(Norm)|**0.0045 ± 0.0010**|0.0028 ± 0.0005|0.0028 ± 0.0005|0.0027 ± 0.0005|0.0028 ± 0.0005|**0.0027 ± 0.0005**|0.0027 ± 0.0005|0.0027 ± 0.0005|0.0027 ± 0.0005|0.0028 ± 0.0005|0.0027 ± 0.0005


* 512x512 img input  

----|**LDCT**|100ep|200ep|300ep|400ep|500ep
----|----|----|----|----|----|----
PSNR|**47.1245 ± 1.7507**|51.0699|51.1611|51.1173|**51.1671**|51.1668
SSIM|**0.9809 ± 0.0093**|0.9893|0.9894|0.9893|0.9893|**0.9894**
RMSE|**18.4390 ± 4.2333**|15.3025|15.1783|15.1788|15.1698|**15.1535**
  
 
<img src="https://github.com/SSinyu/RED_CNN/blob/master/img/LDCT1.PNG" width="400"/><img src="https://github.com/SSinyu/RED_CNN/blob/master/img/REDCNN1.PNG" width="400"/><img src="https://github.com/SSinyu/RED_CNN/blob/master/img/NDCT1.PNG" width="400"/> 

<img src="https://github.com/SSinyu/RED_CNN/blob/master/img/LDCTw.PNG" width="400"/><img src="https://github.com/SSinyu/RED_CNN/blob/master/img/REDCNNw.PNG" width="400"/><img src="https://github.com/SSinyu/RED_CNN/blob/master/img/NDCTw.PNG" width="400"/> 

