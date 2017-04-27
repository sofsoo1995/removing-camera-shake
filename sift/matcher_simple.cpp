#include<vector>
#include<iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include<random>
#include<cmath>
#include<cfloat>


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
template <typename T>
vector<int> argsort(const vector<T> &v) {

  // initialize original index locations
  vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](int i1, int i2) {return v[i1] < v[i2];});

  return idx;
}
//basic algebra
cv::Point2f operator*(cv::Mat M, const cv::Point2f& p)
{ 
    cv::Mat_<double> src(3/*rows*/,1 /* cols */); 

    src(0,0)=p.x; 
    src(1,0)=p.y; 
    src(2,0)=1.0; 

    cv::Mat_<double> dst = M*src; //USE MATRIX ALGEBRA 
    return cv::Point2f(dst(0,0)/dst(2,0),dst(1,0)/dst(2,0)); 
} 

//Combinatory
int logCombi(int n,int k){
  int res_n = 0;
  int res_k =0;
  int res_nk=0;
  for (int i=1; i <= n; i++) {
    res_n += log(i);
    if(i == k) res_k = res_n;
    if(i == (n-k)) res_nk = res_n;
  }
  return res_n - res_k - res_nk;
}

//---------code for ORSA-------------
vector<int> generate_random_sample(int n_sample,vector<DMatch> good_matches){
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> uni(0, good_matches.size());
  vector<int> index;
  int a = n_sample;
  //
  while(a > 0){
    auto random_integer = uni(rng);
    if(find(index.begin(), index.end(), random_integer) == index.end()){
      index.push_back(random_integer);
      a--;
    }
  }
  return index;
}
void bestNFA(vector<double> errors, int n, int n_sample, int alpha0, double &minNFA, int &bestk){
  int result =0;
  double minNFA = DBL_MAX-1; 
  int bestk = 0;
  for(int i =n_sample+1;i<(int)errors.size(); i++){
    double err;
    if(errors[i] == 0) err=-37;
    else err = log(errors[i]);
    double nfa = log(n-n_sample)+logCombi(n,i)+ logCombi(i,n_sample) + (i-n_sample)*(2*err+log(alpha0));
    if(nfa < minNFA){
      minNFA = nfa;
      bestk=i;
    }
  }
}
double symetric_error(Mat_<double> M, Point2f X1, Point2f X2){
  Point2f p= M*X1-X2;
  //cout<<"X1:"<<X1<<endl<<"X2:"<<X2<<endl;
  //cout<<"HX1:"<<M*X1<<endl;
  Point2f p2= M.inv()*X2-X1;
  return sqrt(p.x*p.x + p.y*p.y) + sqrt(p2.x*p2.x + p2.y*p2.y);
}
int homography(vector<int> index, vector<DMatch> good_matches, 
	       vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, Mat &H ){
  
  
  Mat_<double> A(index.size()*2, 9);// matrix to solve
  
  vector<Point2f> list_X;//list of points X
  vector<Point2f> list_Xp;//list of X'
  unsigned int N = index.size();//size
  //centroid for matrix T and T'
  Point2f c(0,0);
  Point2f cp(0,0);
  
  for(unsigned int i=0;i<index.size();i++){
    int b = index[i];
    DMatch match = good_matches[b];
    int queryIdx = match.queryIdx;
    int trainIdx = match.trainIdx;
    KeyPoint xi = keypoints_1[queryIdx];
    KeyPoint xiprime = keypoints_2[trainIdx];
    
    Point2f X = xi.pt;
    Point2f Xprime = xiprime.pt;
    list_X.push_back(X);
    list_Xp.push_back(Xprime);
    c += X;
    cp += Xprime;
    
    //creation of this matrix to find the best homography
    //Point2f test = xi.pt;
    //cout<<"point:" <<test<<endl;
  }


  c = (1.0/N) * c;
  cp = (1.0/N) * cp;
  Mat_<double> T(3,3);//isotropic transfo of X
  Mat_<double> Tp(3,3);// trans for X'
  T = Mat::zeros(3,3,CV_32F);
  Tp = Mat::zeros(3,3, CV_32F);
  double var = 0;
  double varp =0;

    //variance of X and X'
  for (unsigned int i=0; i < index.size(); i++) {
    Point2f v = list_X[i]-c;
    Point2f vp = list_Xp[i]-cp;
    var += v.x*v.x + v.y*v.y;
    varp += vp.x*vp.x + vp.y*vp.y;
  }
  //construction of T
  var = (1.0/(double)N) *var;
  varp = (1.0/(double)N) * varp;
  var = sqrt(var);
  varp = sqrt(varp);
  T.at<double>(0,0)= 1/var;
  T.at<double>(1,1)= 1/var;
  T.at<double>(2,2)= 1;
  T.at<double>(0,2)=-c.x/var;  
  T.at<double>(1,2)= -c.y/var;
  Tp.at<double>(0,0)= 1/varp;
  Tp.at<double>(1,1)= 1/varp;
  Tp.at<double>(2,2)= 1;
  Tp.at<double>(0,2)=-cp.x/varp;  
  Tp.at<double>(1,2)= -cp.y/varp;
  
  for (unsigned int i=0; i < index.size(); i++) {
    Point2f X = (list_X[i] - c)/var;
    Point2f Xprime = (list_Xp[i]-cp)/varp;
    //Point2f X = (list_X[i]);
    //Point2f Xprime = (list_Xp[i]);
    //Construction of A (Ah=0)
    A.at<double>(2*i, 0) = X.x;
    A.at<double>(2*i, 1) = X.y;
    A.at<double>(2*i, 2) = 1;
    A.at<double>(2*i, 3) = 0;
    A.at<double>(2*i, 4) = 0;
    A.at<double>(2*i, 5) = 0;
    A.at<double>(2*i, 6) = -Xprime.x*X.x;
    A.at<double>(2*i, 7) = -Xprime.x*X.y;
    A.at<double>(2*i, 8) = -Xprime.x;
    A.at<double>(2*i+1, 0) = 0;
    A.at<double>(2*i+1, 1) = 0;
    A.at<double>(2*i+1, 2) = 0;
    A.at<double>(2*i+1, 3) = X.x;
    A.at<double>(2*i+1, 4) = X.y;
    A.at<double>(2*i+1, 5) = 1;
    A.at<double>(2*i+1, 6) = -Xprime.y * X.x;
    A.at<double>(2*i+1, 7) = -Xprime.y * X.y;
    A.at<double>(2*i+1, 8) = -Xprime.y;
  }
  
  Mat S, U, Vt, V;
  cv::SVD::compute(A, S, U, Vt, cv::SVD::FULL_UV);
  cv::transpose(Vt, V);
  //cout<<"S"<<S<<endl;
  Mat y = Vt.row(Vt.cols -1);
  y = y.clone();
  H = y.reshape(0,3);
  H = Tp.inv()*H*T;
  double scale = H.at<double>(2,2);
  //cout<<"H"<<endl<<H<<endl<<endl;
  //cout<<"scale:"<<scale<<endl;
  H = H * (1.0/scale);
  
  // *****to check the conditionning of h
  
  //cout<<S.at<double>(0,0)<<endl;
  //cout<<S.at<double>(S.rows-1,0)<<endl;
  return 1;
}
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
    //SIFt to detect descriptors
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptor_1, descriptor_2;
  f2d->detectAndCompute(input, Mat(), keypoints_1, descriptor_1);
  f2d->detectAndCompute(input2,Mat(), keypoints_2, descriptor_2);

  //use Flann to find matches
  FlannBasedMatcher matcher;
  std::vector<DMatch> matches;
  matcher.match(descriptor_1, descriptor_2, matches);
  double max_dist = 0; double min_dist = 10000000;
  for(int i = 0; i < descriptor_1.rows; i++ ){ 
    double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }
  //we take the twenty best matches.
  std::sort(matches.begin(), matches.end(), compareByDistance);
  std::vector< DMatch > good_matches ;
  good_matches.assign(matches.begin(), matches.end());
      
  Mat img_matches;
  drawMatches( input, keypoints_1, input2, keypoints_2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  cv::imwrite(argv[3], img_matches);
  // calcul homography
  

  
  int n_iter=1;
  int n = (int) good_matches.size();
  int n_sample = 4;
  for(int i=0;i<n_iter;i++){
  vector<int> index;
  //1) draw 4 points
  index = generate_random_sample(n_sample, good_matches);
  vector<double> errors;
  Mat_<double> H(3,3);
  //2) homography estimation
  int ratio = homography(index, good_matches, keypoints_1, keypoints_2, H);
  // 3) calculation of errors for all points 
  for(auto m : good_matches){
    int queryIdx = m.queryIdx;
    int trainIdx = m.trainIdx;
    Point2f pt = keypoints_1[queryIdx].pt;
    Point2f pt2 = keypoints_2[trainIdx].pt;
    //cout<<H<<endl;
    errors.push_back(symetric_error(H, pt, pt2));
    cout<<"error :"<< symetric_error(H, pt, pt2)<<endl;
  }
  //4) sorting the error keep track of indexes of the matches
  vector<int> index_match = argsort(errors);
  sort(errors.begin(), errors.end());
  //5) calculation of the best NFA for this homography
  int n = (int)good_matches.size();
  
  }
  
  //creation of matrix
  
  
    // Add results to image and save.
  //cv::Mat output;
  //cv::drawKeypoints(input, keypoints_1, output);
  //cv::imwrite("sift_result.jpg", output);
}
