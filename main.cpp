#include<iostream>
#include<cstdio>
#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;

void deblur(vector<Mat> v, int p, Mat * res);
void displayDft(Mat img);

int main(int argc, char *argv[])
{
  if(argc != 2){
    printf("mauvaise entrée\n");
    return -1;
  }
  vector<Mat> v;//blurred images
  int id =1;
  string name = string(argv[1]);
  Mat img;

  do{
    img = imread(name+std::to_string(id)+".jpg"
		 , 0);
    if(!img.empty())
      v.push_back(img);
    id++;

  }while(!img.empty());

  if(v.size() == 0){
    printf("Error : fichier non valide\n");
    //system("pause");
    return -1;
  }


  namedWindow("resultat",CV_WINDOW_AUTOSIZE);
  Mat u;
  deblur(v, 12, &u);

  //imshow("resultat", u);
  imshow("image", v[0]);
  displayDft(v[0]);
  waitKey(0);
  destroyWindow("resultat");

  return 0;
}


void deblur(vector<Mat> list_vec, int p, Mat * res){

  Mat w(list_vec[0].rows, list_vec[0].cols, CV_32F, Scalar(0));
  Mat up;
  Mat planes[] = {Mat::zeros(list_vec[0].size(), CV_32F), Mat::zeros(list_vec[0].size(), CV_32F)};
  Mat vi_fft;

  Mat wi;
  merge(planes, 2, up);
  for(unsigned int i =0; i<list_vec.size() ; i++){
    //dft of v[i]
    Mat planes[] = {Mat_<float>(list_vec[i]), Mat::zeros(list_vec[i].size(), CV_32F)};

    merge(planes, 2, vi_fft);
    cv::dft(vi_fft, vi_fft);
    cv::split(vi_fft, planes);
    cv::magnitude(planes[0], planes[1], wi);

    // gaussian smoothing
    Mat wi_tmp=wi;
    GaussianBlur( wi_tmp, wi  , Size( 5, 5), 0, 0 );

    //cout << planes[0].size()<<" "<<i<<endl;

    Mat wip; //real type wip
    Mat wip_comp; //complex type wip
    double maxim;
    cv::minMaxLoc(wi,NULL,&maxim,NULL,NULL);
    cv::addWeighted(wi,1/maxim,wi,1/maxim,0,wi);
    cv::pow(wi, p, wip);//wip =|vi|^p
    cout<<wip.at<float>(100, 200)<<endl;
    cv::multiply(wip,planes[0],planes[0]);
    cv::multiply(wip,planes[1],planes[1]);// wip_comp = vi * |vi|^p
    merge(planes, 2, vi_fft);
    cv::add(up,vi_fft, up); //up = up + vi*|vi|^p
    cv::add(w,wip,w);//w = w+ |wi|^p
  }

  split(up, planes);
  cv::divide(planes[0], w, planes[0]);
  cv::divide(planes[1], w, planes[1]);
  merge(planes, 2, up);

  idft(up,*res,DFT_SCALE | DFT_REAL_OUTPUT);
  res -> convertTo(*res, CV_8U);
  imshow("resultat", *res);

}


void displayDft(Mat img){
  Mat complexI;
  Mat realI;
  Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};
  merge(planes, 2, complexI);
  cv::dft(complexI,complexI);
  split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
  magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
  Mat magI = planes[0];

  magI += Scalar::all(1);                    // switch to logarithmic scale
  log(magI, magI);

  //crop the spectrum, if it has an odd number of rows or columns
  magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

  // rearrange the quadrants of Fourier image  so that the origin is at the image center
  int cx = magI.cols/2;
  int cy = magI.rows/2;

  Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
  Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
  Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
  Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

  Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
  q2.copyTo(q1);
  tmp.copyTo(q2);

  idft(complexI,realI,DFT_SCALE | DFT_REAL_OUTPUT);
  realI.convertTo(realI, CV_8U);
  normalize(magI, magI, 0, 1, CV_MINMAX);
  imshow("spectrum magnitude", magI);
  imshow("Reconstructed", realI);
  waitKey();

}
