#include<vector>
#include<iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include<random>
#include"./matcher_simple.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
//basic algebra



int main(){
  Mat input1 = imread("./rotated90.png",0);
  Mat input2 = imread("input_0.png",0);
  Mat output;
  cout<<logCombi(8,5);
  int res = recalage(input1, input2, output);
  cv::imwrite("sift_result.jpg", output);
  //cout<< homogra.inv()*X1-X2<<endl;
}
