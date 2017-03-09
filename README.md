# removing-camera-shake
Removing Camera Shake via Weighted Fourier Burst Accumulation
 How does it work ?
 you execute: g++ `pkg-config --cflags opencv` main.cpp `pkg-config --libs opencv` -o main -std=c++11  
 (no makefile...)
 you do ./main <image>
 where <image> is the common name of all images
 example : ./main anthropologie/REG_00
 there are 8 images

This project is an application of the following article :
https://www.researchgate.net/publication/275964541_Burst_Deblurring_Removing_Camera_Shake_Through_Fourier_Burst_Accumulation
 
