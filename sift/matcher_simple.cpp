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

bool compareByDistance(const DMatch &a, const DMatch &b){
  return a.distance < b.distance;
}

int main(int argc, char *argv[]){
  Ptr<SIFT> f2d = SIFT::create(); 
  if( argc != 4 ){ 
   
    return -1; 
  }
  Mat input = cv::imread(argv[1]); 
  Mat input2 = cv::imread(argv[2]);
    if( !input.data || !input2.data ){ 
      std::cout<< " --(!) Error reading images " << std::endl; 
      return -1; 
    }
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptor_1, descriptor_2;
  f2d->detectAndCompute(input, Mat(), keypoints_1, descriptor_1);
  f2d->detectAndCompute(input2,Mat(), keypoints_2, descriptor_2);
  FlannBasedMatcher matcher;
  std::vector<DMatch> matches;
  matcher.match(descriptor_1, descriptor_2, matches);
  double max_dist = 0; double min_dist = 100;
  for(int i = 0; i < descriptor_1.rows; i++ ){ 
    double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }
  //we take the twenty best matches.
  std::sort(matches.begin(), matches.end(), compareByDistance);
  std::vector< DMatch > good_matches ;
  good_matches.assign(matches.begin(), matches.begin()+20);
      
  Mat img_matches;
  drawMatches( input, keypoints_1, input2, keypoints_2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  cv::imwrite(argv[3], img_matches);
  // calcul homography
  

  //1) Select random points
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> uni(0, good_matches.size());
  vector<int> index;
  int a = 4;
  while(a > 0){
    auto random_integer = uni(rng);
    if(index.find(random_integer) == index.end()){
      index.push_back(random_integer);
      a--;
    }
  }
  vector<KeyPoint> X;
  vector<KeyPoint> Xprime;
  for(auto b : index){
    DMatch match = good_matches[b];
    int queryIdx = match.queryIdx;
    int trainIdx = match.trainIdx;
    KeyPoint xi = keypoints_1[queryIdx];
    KeyPoint xiprime = keypoints_2[trainIdx];
    X.push_back(xi);
    Xprime.push_back(xiprime);
  }

    // Add results to image and save.
  //cv::Mat output;
  //cv::drawKeypoints(input, keypoints_1, output);
  //cv::imwrite("sift_result.jpg", output);
}
