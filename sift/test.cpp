#include<vector>
#include<iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include<random>


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
//basic algebra

cv::Point2f operator*(cv::Mat M, const cv::Point2f& p)
{ 
    cv::Mat_<double> src(3/*rows*/,1 /* cols */); 

    src(0,0)=p.x; 
    src(1,0)=p.y; 
    src(2,0)=1.0; 

    cv::Mat_<double> dst = M*src; //USE MATRIX ALGEBRA 
    return cv::Point2f(dst(0,0),dst(1,0)); 
}

int main(){
  Mat_<double> homogra(2,9);
  homogra.at<double>(0,0)=1.1688;
  homogra.at<double>(0,1)=0.23;
  homogra.at<double>(0,2)=(-62.20);
  homogra.at<double>(0,3)=(-0.013);
  homogra.at<double>(0,4)=1.225;
  homogra.at<double>(0,5)=-6.29;
  homogra.at<double>(0,6)=0;
  homogra.at<double>(0,7)=0;
  homogra.at<double>(0,8)=1;
  homogra.at<double>(1,0)=0.5;
  homogra.at<double>(1,1)=1.2;
  homogra.at<double>(1,2)=1;
  homogra.at<double>(1,3)=1.1688;
  homogra.at<double>(1,4)=0.23;
  homogra.at<double>(1,5)=(-62.20);
  homogra.at<double>(1,6)=(-0.013);
  homogra.at<double>(1,7)=1.225;
  homogra.at<double>(1,8)=-6.29;
  Point2f X1(7,4);
  Point2f X2(1,0);
  cv::Mat S, U, Vt;
  cv::SVD::compute(homogra, S, U, Vt, cv::SVD::FULL_UV);
  std::cout << "U" << std::endl << U << std::endl << std::endl;
  Mat v; 
  transpose(Vt,v);
  cout<<"v"<<endl<<v<<endl;
  Mat y = v.col(v.cols-1);
  y = y.clone();
  cout<< "col : "<< y<<endl;
  Mat h = y.reshape(0,3);
  cout<<endl<<"h:"<<h<<endl;
  cv::Mat Ut;
  cv::transpose( U, Ut );
  std::cout << "U * Ut" << std::endl << U * Ut << std::endl;
  
  //cout<< homogra.inv()*X1-X2<<endl;
}
