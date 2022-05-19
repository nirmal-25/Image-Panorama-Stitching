# Image Stitching and Warping

2D Homography is implemented from scratch here; note that there is a ```findHomography()``` function in OpenCV. Then warping is done to map the source image to the destination using the inverse of the Homography matrix obtained. 

Corresponding keypoints between images are found using SIFT and further refined by the RANSAC algorithm

The ```.ipynb``` file can be used for panorama stitching of custom images, along with the examples given. For details on the image warping implementation refer to ```image_warping.py``` 

#### Results

*Image Warping* 

<img src="https://user-images.githubusercontent.com/51696913/169203052-f6f3d11e-cda6-4c4b-99f5-031b376b4156.png" width="400">

*RANSAC using SIFT keypoints*

<img src="https://user-images.githubusercontent.com/51696913/169204171-9b61072c-215a-46dd-99ca-c2a11c0237b0.png" width="400">

*Sample output images*

<img src="https://user-images.githubusercontent.com/51696913/169203134-8c3820c2-90fb-4eae-ace2-c50ba6f4b0cd.png" width="400">

<img src="https://user-images.githubusercontent.com/51696913/169203115-da6e417e-1c4d-4600-90c9-21902c9f22a2.png" width="400">

<img src="https://user-images.githubusercontent.com/51696913/169203117-f3a21fe2-af55-47ba-bcfa-4a2c5b18301a.png" width="400">
