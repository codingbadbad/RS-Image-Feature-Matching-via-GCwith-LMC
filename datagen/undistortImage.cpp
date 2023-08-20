#include <opencv2/opencv.hpp>
#include <string>
using namespace std;
using namespace cv;

vector<int> find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {

  Mat descriptors_1, descriptors_2;


      Ptr<SIFT> sift= SIFT::create(3000);


    Mat ori_des, dst_des;
    sift->detectAndCompute(img_1, noArray(), keypoints_1, ori_des);
    sift->detectAndCompute(img_2, noArray(), keypoints_2, dst_des);

    Ptr<BFMatcher> bf = BFMatcher::create(NORM_L2, true);

    vector<DMatch> bf_matches;
    bf->match(ori_des, dst_des, bf_matches);


      matches = bf_matches;


  double min_dist = 10000, max_dist = 0;

  // for (int i = 0; i < descriptors_1.rows; i++) {
  //   double dist = match[i].distance;
  //   if (dist < min_dist) min_dist = dist;
  //   if (dist > max_dist) max_dist = dist;
  // }

  // printf("-- Max dist : %f \n", max_dist);
  // printf("-- Min dist : %f \n", min_dist);
  vector<int> right(matches.size(),0);
 

  return right;
}
int main(int argc, char **argv) {
  // distortion_coeffs: [-0.017811595366268803, 0.04897078939103475, -0.041363300782847834,
  //   0.011440891936886532]

  // intrinsics: [275.3385453506587, 275.0852058534152, 315.7697752181792, 233.72625444124952]

  double k1 = -0.017811595366268803, k2 =  0.04897078939103475, p1 = -0.041363300782847834, p2 =  0.011440891936886532;

  double fx = 275.3385453506587, fy = 275.0852058534152, cx =  315.7697752181792, cy = 233.72625444124952;

  cv::Mat image = cv::imread(argv[1]);

  int rows = image.rows, cols = image.cols;
  cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);   // 去畸变以后的图

  cv::Mat insmatrix=(cv::Mat_<double>(3,3)<<fx,0,cx,0,fy,cy,0,0,1);



  cv::Mat image_undistort2=cv::Mat(rows,cols,CV_8UC1);
  cv::Mat vec=(cv::Mat_<double>(4,1)<<k1,k2,p1,p2);


cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(insmatrix, vec, image.size(), 1, image.size(), 0, 1);

cv::fisheye::undistortImage(image, image_undistort2, insmatrix, vec, newCameraMatrix);



  // cv::getOptimalNewCameraMatrix(insmatrix,vec,image.size(),1,image.size(),0 , 1);


  // cv::fisheye::undistortImage(image,image_undistort2,insmatrix,vec,
  //                     cv::getOptimalNewCameraMatrix(insmatrix,vec,image.size(),1,image.size(), 0));
  // cv::fisheye::undistortImage(image,image_undistort2,insmatrix,vec);


  cv::imshow("distorted", image);

cv::imshow("undistorted2", image_undistort2);

  cv::imwrite("undistorted.jpg", image_undistort);
  cv::waitKey();
  return 0;
}
