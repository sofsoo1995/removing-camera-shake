
#ifndef ADD_H
#define ADD_H

#include<vector>
#include<iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
double logCombi(int n,int k);
double homography(vector<int> index, const vector<DMatch> good_matches, 
		    const vector<KeyPoint> keypoints_1, const vector<KeyPoint> keypoints_2, Mat &H );
int recalage(Mat input, Mat input2, Mat &output);
#endif
