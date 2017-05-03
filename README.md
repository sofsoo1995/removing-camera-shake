# removing-camera-shake
Removing Camera Shake via Weighted Fourier Burst Accumulation
 How does it work ?
 you execute: g++ `pkg-config --cflags opencv` main.cpp sift/matcher_simple.ccp `pkg-config --libs opencv` -o main -std=c++11  
 (no makefile...)
 you do ./main <image> <extension>
 where <image> is the common name of all images
 example : ./main anthropologie/REG_00 jpg
 there are 8 images


SIFT
important for me :

"LD_LIBRARY_PATH=/usr/local/lib"

"export LD_LIBRARY_PATH"

"g++ -L-I/usr/local/lib `pkg-config --cflags opencv` matcher_simple.cpp `pkg-config --libs opencv` -o match -std=c++11"

This project is an application of the following article :
https://www.researchgate.net/publication/275964541_Burst_Deblurring_Removing_Camera_Shake_Through_Fourier_Burst_Accumulation
