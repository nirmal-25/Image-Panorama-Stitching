### Image Stitching and Warping

2D Homography is implemented from scratch here; note that there is a findHomography() function in OpenCV. Then warping is done to map the source image to the destination using the inverse of the Homography matrix obtained. 
Corresponding keypoints between images are found using SIFT and further refined by the RANSAC algorithm
The ```.ipynb``` file can be used for panorama stitching of custom images, along with the examples given. The .py file 
