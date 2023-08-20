//edit by yanchun.liu in May 2023 
// sorry for my worse coding style


#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio> 
#include <fstream>
#include <sstream>
using namespace std;
using namespace cv;
#include <iostream>
#include <fstream>
#include <vector>
#include <utility> 
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <opencv2/imgproc/imgproc.hpp>
// #include "extra.h" // use this if in OpenCV2
#include <fstream>
#include <sstream>
using namespace std;
using namespace cv;
#include <iostream>
#include <fstream>
#include <vector>
#include <utility> // for std::pair
#include <opencv2/core.hpp>
#include <dirent.h>





int split(string name, int i ,Mat& croppedImage1,Mat& croppedImage2) {

    cv::Mat originalImage = cv::imread(name);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distributionWidth(300, 800);
    std::uniform_int_distribution<int> distributionHeight(300, 800);
    std::uniform_real_distribution<float> distributionRotation(0.0f, 1.0f);

    // 随机生成第一张图像的宽度和高度
    int cropWidth1 = distributionWidth(generator);
    int cropHeight1 = distributionHeight(generator);
    // int cropHeight1 = cropWidth1;

    // 随机生成第一张图像的起始坐标
    int x1 = std::uniform_int_distribution<int>(0, originalImage.cols - cropWidth1 -1)(generator);
    int y1 = std::uniform_int_distribution<int>(0, originalImage.rows - cropHeight1 -1)(generator);


    // 随机生成第二张图像的宽度和高度
    int cropWidth2 = cropWidth1;
    int cropHeight2 = cropHeight1;

    // int cropWidth2 = distributionHeight(generator);
    // int cropHeight2 = distributionHeight(generator);

    // 随机生成第二张图像的起始坐标，确保与第一张图像有一定的交集
    int x2, y2;
    float intersectionThreshold = 0.10f;  // 5%的面积相交阈值
    float intersectionArea = 0.0f;
    do {

x2 = std::uniform_int_distribution<int>(std::max(0, x1 - cropWidth2), std::min(originalImage.cols - cropWidth2, x1 + cropWidth1))(generator);
y2 = std::uniform_int_distribution<int>(std::max(0, y1 - cropHeight2), std::min(originalImage.rows - cropHeight2, y1 + cropHeight1))(generator);

        // 计算交集面积
        int intersectionX = std::max(x1, x2);
        int intersectionY = std::max(y1, y2);
        int intersectionWidth = std::min(x1 + cropWidth1, x2 + cropWidth2) - intersectionX;
        int intersectionHeight = std::min(y1 + cropHeight1, y2 + cropHeight2) - intersectionY;
        intersectionArea = intersectionWidth * intersectionHeight;
    } while (intersectionArea < intersectionThreshold * (cropWidth2 * cropHeight2));

    // 裁剪图像1
    cv::Rect cropRect1(x1, y1, cropWidth1, cropHeight1);
    croppedImage1 = originalImage(cropRect1).clone();

    // 裁剪图像2
    cv::Rect cropRect2(x2, y2, cropWidth2, cropHeight2);
    croppedImage2 = originalImage(cropRect2).clone();

    // 对40%的图像进行随机旋转
    float rotationThreshold = 0.4f;
    float randomRotation = distributionRotation(generator);
    
// 随机选择要旋转的图像
bool rotateFirstImage = std::uniform_int_distribution<int>(0, 1)(generator) == 0;

if (randomRotation < rotationThreshold) {
    float angle = std::uniform_real_distribution<float>(-120.0f, 120.0f)(generator);

    if (rotateFirstImage) {
        cv::Point2f center1(croppedImage1.cols / 2.0f, croppedImage1.rows / 2.0f);
        cv::Mat rotationMatrix1 = cv::getRotationMatrix2D(center1, angle, 1.0);

        // 计算旋转后的图像大小
        cv::Rect bbox = cv::RotatedRect(center1, croppedImage1.size(), angle).boundingRect();

        // 平移图像
        rotationMatrix1.at<double>(0, 2) += bbox.width / 2.0 - center1.x;
        rotationMatrix1.at<double>(1, 2) += bbox.height / 2.0 - center1.y;

        cv::warpAffine(croppedImage1, croppedImage1, rotationMatrix1, bbox.size(),1,0,
         cv::Scalar(255,255,255));
    } else {
        cv::Point2f center2(croppedImage2.cols / 2.0f, croppedImage2.rows / 2.0f);
        cv::Mat rotationMatrix2 = cv::getRotationMatrix2D(center2, angle, 1.0);

        // 计算旋转后的图像大小
        cv::Rect bbox = cv::RotatedRect(center2, croppedImage2.size(), angle).boundingRect();

        // 平移图像
        rotationMatrix2.at<double>(0, 2) += bbox.width / 2.0 - center2.x;
        rotationMatrix2.at<double>(1, 2) += bbox.height / 2.0 - center2.y;

        cv::warpAffine(croppedImage2, croppedImage2, rotationMatrix2, bbox.size(),1,0,
         cv::Scalar(255,255,255));
    }
}


// 随机确定是否对图像进行缩放
float scaleThreshold = 0.4f;
float randomScale = distributionRotation(generator);

if (randomScale < scaleThreshold) {
    // 随机生成缩放比例
    float scale = std::uniform_real_distribution<float>(0.5f, 1.0f)(generator);

    // 随机选择要缩放的图像
    bool scaleFirstImage = std::uniform_int_distribution<int>(0, 1)(generator) == 0;

    // 根据缩放比例调整图像大小
    if (scaleFirstImage) {
        cv::resize(croppedImage1, croppedImage1, cv::Size(), scale, scale);
    } else {
        cv::resize(croppedImage2, croppedImage2, cv::Size(), scale, scale);
    }
}

// 显示裁剪后的图像
cv::imwrite("./out/"+to_string(i)+"l.png", croppedImage1);
cv::imwrite("./out/"+to_string(i)+"r.png", croppedImage2);
// cv::waitKey(0);

return 0;
}












std::vector<std::string> readFolder(const std::string& folderPath) {
    std::vector<std::string> fileNames;
    DIR* dir;
    struct dirent* entry;

    dir = opendir(folderPath.c_str());
    if (dir) {
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_type == DT_REG) {
                fileNames.push_back(entry->d_name);
            }
        }
        closedir(dir);
    }

    return fileNames;
}


std::string padZeros(const std::string& number) {
    std::string paddedNumber = number;
    while (paddedNumber.length() < 8) {
        paddedNumber = "0" + paddedNumber;
    }
    return paddedNumber+".jpg";
}


//  给自己找麻烦
int GETf(string inputname,
        std::vector<std::pair<int, int>> &fileNames ,
        std::vector<cv::Mat>& matrices ) {
    std::ifstream inputFile( inputname);


    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        std::string fileNamePrefix;
        std::string fileName;
        double value;
        string num1, num2;
        // 读取文件名的前两个数字
        iss >> num1;
        iss >> num2;
        int num11 = atoi(num1.c_str());
        int num22 = atoi(num2.c_str());



        // 将文件名前两个数字转换为八位数字格式
        std::string png1 = padZeros(num1);
        std::string png2 = padZeros(num2);

        // 读取文件名的后六位数字
        // fileName = fileNamePrefix.substr(2, 6);

        // 读取矩阵的九个值
        cv::Mat F(3, 3, CV_64F);
        for (int i = 0; i < 9; i++) {
            iss >> value;

            F.at<double>(i / 3, i % 3) = value;
        }

        fileNames.emplace_back(num11, num22);
        matrices.push_back(F);
    }

    inputFile.close();



    return 0;
}


std::vector<cv::Mat> matrix(string inputname) {
    std::ifstream inputFile(inputname);
    std::vector<std::vector<double>> data; 

    double value;
    while (inputFile >> value) {
        std::vector<double> row; 
        row.push_back(value);
        for (int i = 1; i < 9; i++) {
            inputFile >> value;
            row.push_back(value);
        }
        data.push_back(row);
    }

    inputFile.close();


    // for (const auto& row : data) {
    //     for (const auto& val : row) {
    //         std::cout << val << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // ins
    std::vector<cv::Mat> internalMatrices;
    for (const auto& row : data) {
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0, 0) = row[0];
        K.at<double>(0, 2) = row[2];
        K.at<double>(1, 1) = row[4];
        K.at<double>(1, 2) = row[5];
        internalMatrices.push_back(K);
    }



    return internalMatrices;
}
float euclideanDistance(const Point2f& p1, const Point2f& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return sqrt(dx*dx + dy*dy);
}


vector <int> find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

int pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                          std::vector<KeyPoint> keypoints_2,
                          std::vector<DMatch> matches,
                          vector<int> &mask) {

if (matches.size()<30)
{
  return 0 ;
}



  vector<Point2f> points1;
  vector<Point2f> points2;

  for (int i = 0; i < (int) matches.size(); i++) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

 

  

 
  Mat H;
  H = findHomography(points1, points2, RANSAC, 10 ,noArray(), 500);
  // cout << "homography_matrix is " << endl << homography_matrix << endl;

    Mat Hfloat = (cv::Mat_<float>(3,3)<<  static_cast<float> (H.at<double>(0,0)),  static_cast<float> (H.at<double>(0,1)) , static_cast<float> (H.at<double>(0,2))  ,
                                          static_cast<float> (H.at<double>(1,0)),  static_cast<float> (H.at<double>(1,1)) , static_cast<float> (H.at<double>(1,2))  , 
                                          static_cast<float> (H.at<double>(2,0)),  static_cast<float> (H.at<double>(2,1)) , static_cast<float> (H.at<double>(2,2))  
                                      );

        float threshold = 4.0f;  
        
        for (int i = 0; i < matches.size(); i++) {
                DMatch bestMatch = matches[i];
                Point2f p1 = keypoints_1[bestMatch.queryIdx].pt;
                Point2f p2 = keypoints_2[bestMatch.trainIdx].pt;
                Mat pt1 = (cv::Mat_<float>(3,1) <<p1.x, p1.y, 1.0f);
                Mat pt2 = (cv::Mat_<float>(3,1) <<p2.x, p2.y, 1.0f);
                // Point3f pt1(p1.x, p1.y, 1.0f), pt2(p2.x, p2.y, 1.0f);
                // Mat pt1Mat(pt1), pt2Mat(pt2);
                Mat transformedPt1Mat = Hfloat * pt1  ;
                Point2f transformedPt1(transformedPt1Mat.at<float>(0, 0) / transformedPt1Mat.at<float>(2, 0),
                                       transformedPt1Mat.at<float>(1, 0) / transformedPt1Mat.at<float>(2, 0));

                // cout<<transformedPt1<< " " <<p2<<endl;
                float distance = euclideanDistance(transformedPt1, p2);
                if (distance < threshold) {
      
                    mask.push_back(1);
                } else {
      
                    mask.push_back(0);

                }









}
  int count = 0;
  for (int i = 0; i < mask.size(); i++)
  {
    if (mask[i]==1) count++;
    // cout<<static_cast<int>(mask[i])<<endl;
  }

return count;
}

Point2d pixel2cam(const Point2f &p, const Mat &K);

int datagen(const vector<DMatch> bf_matches ,
            const vector<KeyPoint> ori_kp,
            const vector<KeyPoint> dst_kp,
            vector<int>  calc,
            Mat ori_img,
            Mat dst_img
            );

//  arg1  camera ins;  agr2  F;arg3 num
int single(string camerains,string funda,int th 
          ) {




      std::vector<std::pair<int, int>> fileNames ;
      std::vector<cv::Mat>  F;

    std::vector<cv::Mat> INS =  matrix(camerains);
    GETf(funda,fileNames,F);


  Mat Fun = (Mat_<float>(3, 3) <<  static_cast<float> (F[th].at<double>(0,0)), static_cast<float> ( F[th].at<double>(0,1) ), static_cast<float> (F[th].at<double>(0,2) ) ,
                                   static_cast<float> (F[th].at<double>(1,0)), static_cast<float> ( F[th].at<double>(1,1) ), static_cast<float> (F[th].at<double>(1,2) ) , 
                                   static_cast<float> (F[th].at<double>(2,0)), static_cast<float> ( F[th].at<double>(2,1) ), static_cast<float> (F[th].at<double>(2,2) ) 
                                      );
  cout << "fun  " << endl << Fun << endl;
  
  Mat K = (Mat_<float>(3, 3) <<    static_cast<float> (INS[th].at<double>(0,0)), static_cast<float> ( INS[th].at<double>(0,1) ),static_cast<float> ( INS[th].at<double>(0,2)) ,
                                   static_cast<float> (INS[th].at<double>(1,0)), static_cast<float> ( INS[th].at<double>(1,1) ),static_cast<float> ( INS[th].at<double>(1,2)) , 
                                   static_cast<float> (INS[th].at<double>(2,0)), static_cast<float> ( INS[th].at<double>(2,1) ),static_cast<float> ( INS[th].at<double>(2,2)) 
                                      );


   cout<<K.type()<<endl; 




int f1 = fileNames[th].first;
int f2 = fileNames[th].second;

cout<<padZeros(to_string(f1))<<endl;
cout<<padZeros(to_string(f2))<<endl;

  Mat img_1 = imread("./Images/"+padZeros(to_string(f1)));
  Mat img_2 = imread("./Images/"+padZeros(to_string(f2)));
  assert(img_1.data && img_2.data && "Can not load images!");


  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  vector<int> right  = find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;


  //-- 估计两张图像间运动
  Mat R, t;

  vector<Point2f> points1;
  vector<Point2f> points2;
  for (int i = 0; i < (int) matches.size(); i++) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }
  Mat computF;
    vector<unsigned char> mask;

  computF = findEssentialMat(points1,points2,K,RANSAC,
                                     0.999, 2,
                                     mask);



vector<int> maskConverted;
maskConverted.reserve(mask.size());

for (auto value : mask) {
  maskConverted.push_back(static_cast<int>(value));
}
     int inliner = 0;
  vector<DMatch> maskmatch;                                     
  for (int i = 0; i < maskConverted.size(); i++)
  {
    // cout<< maskConverted[i] <<endl;;
    if (maskConverted[i] == true)
    {
        inliner++;
     maskmatch.push_back(matches[i]);
    }
  }
  Mat maskmat;
        drawMatches(img_1,keypoints_1,img_2,keypoints_2,maskmatch,maskmat,Scalar(255,0,0),Scalar(0,0,255));
      resize(maskmat,maskmat,Size(maskmat.cols*0.75,maskmat.rows*0.75));
      imshow("ransacmask"+to_string(maskmatch.size()),maskmat);
      waitKey(0);






 
  int i = 0;
  
  int small = 0;

  vector<DMatch> matchfilter;
    Mat match;

  for (int j = 0 ; j< matches.size() ; j++) {


    Mat y1 = (Mat_<float>(3, 1) <<static_cast<float>( keypoints_1[matches[j].queryIdx].pt.x)
                              , static_cast<float>(keypoints_1[matches[j].queryIdx].pt.y), float(1));


    Mat y2 = (Mat_<float>(3, 1) <<static_cast<float>( keypoints_2[matches[j].trainIdx].pt.x), 
                              static_cast<float>(keypoints_2[matches[j].trainIdx].pt.y), float(1));

    // Point2f pt1 = pixel2cam(  keypoints_1[matches[j].queryIdx].pt, K);

    // Mat y1 = (Mat_<float>(3, 1) << static_cast<float>(pt1.x), static_cast<float>(pt1.y), 1);


    // Point2f pt2 = pixel2cam(Point2d(keypoints_2[matches[j].trainIdx].pt), K);
    // Mat y2 = (Mat_<float>(3, 1) << static_cast<float>(pt2.x), static_cast<float>(pt2.y), 1);
    // cout<<y1 <<y2<<endl;



    Mat d1 = y2.t() * Fun * y1;

    Mat line1 =  Fun * y1;
    double eva1 =     abs(d1.at<float>(0,0)) / sqrt(pow(line1.at<float>(0,0),2)+pow(line1.at<float>(1,0),2)) ;

    Mat line2 = y2.t() *Fun.inv().t();
    // Mat line2 = y2.t() *Fun.t();
    double eva2 =   abs(d1.at<float>(0,0)) / sqrt(pow(line2.at<float>(0,0),2)+pow(line2.at<float>(1,0),2)) ;

    cout << "epipolar constraint = "<<d1.at<float>(0,0) <<" :"<<eva1 << "::" << eva2<<endl;

    if( eva1 <35  
    // && (  eva2 <eva1*20  || 
      // && (eva2<3e6)
      ){ 
        small ++;
      matchfilter.push_back(matches[j]);
    }
  }


  cout<< small << " / "<< matches.size()<<endl;
  cout<< maskmatch.size() << " / "<< matches.size()<<endl;



/// rename 
    string name = "./for_single_graph/" + to_string(inliner) +"from" + 
                        to_string(f1) + to_string(f2)+"distance"+to_string(small) +".csv";
    //  只能初始化const char×
    const char* newname = name.c_str();
    datagen(matchfilter,keypoints_1,keypoints_2,maskConverted,img_1,img_2 );
    int result = std::rename("test.csv",newname);





      drawMatches(img_1,keypoints_1,img_2,keypoints_2,matchfilter,match,Scalar(255,0,0),Scalar(255,255,255));
      if( match.cols > 800)  resize(match,match,Size(match.cols*0.75,match.rows*0.75));
      imshow("distance"+to_string(small),match);
      waitKey(0);

  return 0;
}


















vector<int> find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
  //-- 初始化
  Mat descriptors_1, descriptors_2;


      Ptr<SIFT> sift= SIFT::create(3000);


    Mat ori_des, dst_des;
    sift->detectAndCompute(img_1, noArray(), keypoints_1, ori_des);
    sift->detectAndCompute(img_2, noArray(), keypoints_2, dst_des);

    // Create a Brute-Force matcher
    Ptr<BFMatcher> bf = BFMatcher::create(NORM_L2, true);

    // Match the feature descriptors
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

Point2d    pixel2cam(const Point2f &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<float>(0, 2)) / K.at<float>(0, 0),
      (p.y - K.at<float>(1, 2)) / K.at<float>(1, 1)
    );
}

vector<int> find_KNN(int K, vector<vector<double>> &distance_matrix, int index) {
    vector<double> distance_vector = distance_matrix[index]; // extract the distance vector of the i-th node
    vector<pair<double, int>> distance_index(distance_vector.size());
    for (size_t i = 0; i < distance_vector.size(); ++i) {
        distance_index[i] = make_pair(distance_vector[i], i);
    }
    sort(distance_index.begin(), distance_index.end());
    vector<int> K_neighbor_index(K + 1);
    for (int i = 0; i <= K; ++i) {
        K_neighbor_index[i] = distance_index[i].second;
    }
    return K_neighbor_index;
}
vector<vector<double>> distance_cal(vector<Point2f> &cord_list) {
    int num = cord_list.size(); 
    vector<vector<double>> distance_matrix(num, vector<double>(num, 0)); 
    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < num; ++j) {
            // calculate the Euclidean distance
            double distance = sqrt(pow(cord_list[i].x - cord_list[j].x, 2) + pow(cord_list[i].y - cord_list[j].y, 2));
            if (i != j) {

                distance_matrix[i][j] = distance + 0.0000001;
            }
        }
    }
    return distance_matrix;
}







int datagen(const vector<DMatch> bf_matches ,
            const vector<KeyPoint> ori_kp,
            const vector<KeyPoint> dst_kp,
            vector<int>  calc,
            Mat ori_img,
            Mat dst_img
           ){

    ofstream ofs;
    ofs.open("test.csv", ios::out);



    vector<Point2f> ori_cord_list;
    for (const auto &m : bf_matches) {
        ori_cord_list.push_back(ori_kp[m.queryIdx].pt);
    }




    for (int i = 0 ; i<bf_matches.size() ; i++) 
    {


        // Point2f o1 = kp1l;
        // Point2f o2 = kp1r;

        // kp1l = kp1l - Point2f(ori_img.cols/2,ori_img.rows/2);
        // kp1r = kp1r - Point2f(ori_img.cols/2,ori_img.rows/2);
        
        // kp1l.x = kp1l.x / ori_img.cols;
        // kp1l.y = kp1l.y / ori_img.rows;
        // kp1r.x = kp1r.x / ori_img.cols;
        // kp1r.y = kp1r.y / ori_img.rows;


 vector<vector<double>> ori_dis_mat = distance_cal(ori_cord_list);

        vector<int> nibor=  find_KNN(30, ori_dis_mat, i);
        for (int j= 0 ; j < nibor.size() ; j++){    /// nibor  in  hole img
                    Point2f kp1l = ori_kp[bf_matches[nibor[j]].queryIdx].pt;
                    Point2f kp1r = dst_kp[bf_matches[nibor[j]].trainIdx].pt;

                    Point2f kp_centerl = ori_kp[bf_matches[nibor[0]].queryIdx].pt;
                    Point2f kp_centerr = dst_kp[bf_matches[nibor[0]].trainIdx].pt;


                    int cols = std::max(ori_img.cols, dst_img.cols);
                    int rows = std::max(ori_img.rows, dst_img.rows);

                    double wradio = static_cast<double> (ori_img.cols)/ static_cast<double> (dst_img.cols);
                    double hradio = static_cast<double> (ori_img.rows)/ static_cast<double> (dst_img.rows);


                    double maxnorm =   max(cv::norm(Point2f(kp_centerr.x  - kp_centerl.x,
                                                            kp_centerr.y  - kp_centerl.y ))  ,
                                                  cv::norm(Point2f(kp1r.x -kp1l.x,
                                                                   kp1r.y -kp1l.y)));

                    float angle = cv::fastAtan2(kp_centerr.y - kp_centerl.y, 
                                           kp_centerr.x - kp_centerl.x) - 
                                    
                                    cv::fastAtan2((kp1r.y -kp1l.y),
                                                (kp1r.x  - kp1l.x));
                                                
                    double denominator = std::max(cv::norm(Point2d(kp1l.x - kp_centerl.x, kp1l.y - kp_centerl.y)),
                              cv::norm(Point2d(kp1r.x - kp_centerr.x, kp1r.y - kp_centerr.y)));
                    double result = 0;
                    if (denominator > 0) {
                        double numerator = std::min(cv::norm(Point2d(kp1l.x - kp_centerl.x, kp1l.y - kp_centerl.y)),
                                                    cv::norm(Point2d(kp1r.x - kp_centerr.x, kp1r.y - kp_centerr.y)));
                        result = numerator / denominator;

                    } else {
                            result=1;
                    }

                          if (abs(angle) < 90) angle = 1 - abs(angle)/ 90; else angle = 0;;
                          // angle = 1 - abs(angle)/ 180;
                            if(calc[i]==1) {ofs<<1<<','; }else {
    
                                ofs <<0<<',';  
                                // cout <<" 0:"<< o1<<o2 << kp1l << kp1r << endl;
                            }
                            ofs << i <<','<<nibor[j]
                        // // 中心节点
                        //  <<','<<kp_centerl.x/ ori_img.cols
                        //  <<','<<kp_centerl.y/ ori_img.rows 
                        //  <<','<<kp_centerr.x/ dst_img.cols                             
                        //  <<','<<kp_centerr.y/ dst_img.rows   

                        //  //邻居的节点
                        //  <<','<<kp1l.x      / ori_img.cols                        
                        //  <<','<<kp1l.y      / ori_img.rows
                        //  <<','<<kp1r.x      / dst_img.cols                             
                        //  <<','<<kp1r.y      / dst_img.rows
                        // 中心节点
                         <<','<<kp_centerl.x / cols
                         <<','<<kp_centerl.y / rows 
                         <<','<<kp_centerr.x / cols                             
                         <<','<<kp_centerr.y / rows   

                        // 中心节点的运动向量

                        //  //邻居的节点
                         <<','<<kp1l.x  / cols                        
                         <<','<<kp1l.y  / rows
                         <<','<<kp1r.x  / cols                             
                         <<','<<kp1r.y  / rows
                        // // 中心节点的运动向量
                        //  <<','<<kp_centerr.x/ dst_img.cols   - kp_centerl.x/ ori_img.cols
                        //  <<','<<kp_centerr.y/ dst_img.rows   - kp_centerl.y/ ori_img.rows  

                        //  //邻居的运动向量
                        //  <<','<< kp1r.x/ dst_img.cols   - kp1l.x /  ori_img.cols
                        //  <<','<< kp1r.y/ dst_img.rows    -kp1l.y /  ori_img.rows

                        // // 中心节点的运动向量
                         <<','<<kp_centerr.x/ cols - kp_centerl.x/cols
                         <<','<<kp_centerr.y/ rows - kp_centerl.y/rows  

                         //邻居的运动向量
                         <<','<< (kp1r.x/cols -kp1l.x/cols)
                         <<','<< (kp1r.y/rows -kp1l.y/rows)
                        // // // 运动向量 一致性
                        //  <<','<< (Point2f(kp_centerr.x/ori_img. cols - kp_centerl.x/ ori_img.cols,
                        //                   kp_centerr.y/ori_img.rows - kp_centerl.y/ ori_img.rows )/maxnorm).dot(
                        //           (Point2f(kp1r.x/ ori_img.cols -kp1l.x/ ori_img.cols,
                        //                    kp1r.y/ ori_img.rows -kp1l.y/ ori_img.rows)/maxnorm    )) 

                        // 运动向量 一致性
                         <<','<< (Point2f(kp_centerr.x - kp_centerl.x,
                                          kp_centerr.y - kp_centerl.y )/maxnorm).dot(
                                  (Point2f(kp1r.x -kp1l.x,
                                           kp1r.y -kp1l.y)/maxnorm    )) 

                        //  // 邻居与中心节点的莫长
                        //  <<","<<sqrt(pow(kp1l.x/ ori_img.cols - kp_centerl.x/ori_img.cols,2)
                        //             +pow(kp1l.y/ ori_img.rows - kp_centerl.y/ori_img.rows,2))
                         
                        //  <<","<<sqrt(pow(kp1r.x/ dst_img.cols - kp_centerr.x/dst_img.cols ,2)
                        //             +pow(kp1r.y/ dst_img.rows - kp_centerr.y/dst_img.rows ,2))
                        //   // 莫长之差
                        // <<","<< abs(
                        //         sqrt(pow(kp1l.x/ ori_img.cols - kp_centerl.x/ori_img.cols,2)
                        //             +pow(kp1l.y/ ori_img.rows - kp_centerl.y/ori_img.rows,2))

                        //        -sqrt(pow(kp1r.x/dst_img.cols - kp_centerr.x/ dst_img.cols ,2)
                        //             +pow(kp1r.y/dst_img.rows - kp_centerr.y/ dst_img.rows ,2))
                        //             )




                        // <<","<< std::min(cv::norm(Point2f(kp_centerr.x*wradio/ cols - kp_centerl.x/cols,
                        //                                   kp_centerr.y*hradio/ rows - kp_centerl.y/rows  )), 
                        //                 cv::norm(Point2f((kp1r.x*wradio/cols -kp1l.x/cols),
                        //                 (kp1r.y*hradio/rows -kp1l.y/rows)))) / std::max(cv::norm(Point2f(kp_centerr.x*wradio/ cols - kp_centerl.x/cols,
                        //                           kp_centerr.y*hradio/ rows - kp_centerl.y/rows  )), 
                        //                 cv::norm(Point2f((kp1r.x*wradio/cols -kp1l.x/cols),
                        //                 (kp1r.y*hradio/rows -kp1l.y/rows))))  




                        //  // 邻居与中心节点的莫长之比
                        <<","<<std::min(   cv::norm(  Point2f(kp_centerr.x - kp_centerl.x,kp_centerr.y - kp_centerl.y )),
                                           cv::norm( (Point2f(kp1r.x -kp1l.x,  kp1r.y -kp1l.y)  ) ))
                                           /std::max(   cv::norm(  Point2f(kp_centerr.x - kp_centerl.x,kp_centerr.y - kp_centerl.y )),
                                                        cv::norm( (Point2f(kp1r.x -kp1l.x,  kp1r.y -kp1l.y)  ) ))



                        <<","<< angle


                        // //  邻居与中心节点的莫长
                        //  <<","<<sqrt(pow(kp1l.x/ cols - kp_centerl.x/cols,2)
                        //             +pow(kp1l.y/ rows - kp_centerl.y/rows,2))
                         
                        //  <<","<<sqrt(pow(kp1r.x*wradio/ cols - kp_centerr.x*wradio/cols ,2)
                        //             +pow(kp1r.y*hradio/ rows - kp_centerr.y*hradio/rows ,2))
                          // 莫长之差
                        // <<","<< abs(
                        //             sqrt(pow(kp1l.x/ cols - kp_centerl.x/cols,2)
                        //                 +pow(kp1l.y/ rows - kp_centerl.y/rows,2))
                        //             -sqrt(pow(kp1r.x/ cols - kp_centerr.x/ cols ,2)
                        //                  +pow(kp1r.y/ rows - kp_centerr.y/ rows ,2))   )

                        <<","<< result
                                    
                         <<',';
                // kp2l = kp2l - Point2f(ori_img.cols/2,ori_img.rows/2);
                // kp2r = kp2r - Point2f(ori_img.cols/2,ori_img.rows/2);
                for (int k= 0 ; k < nibor.size() ; k++){

                    Point2f kp2l = ori_kp[bf_matches[nibor[k]].queryIdx].pt;


                    Point2f kp2r = dst_kp[bf_matches[nibor[k]].trainIdx].pt;

                    Point2f v1 = kp1r - kp1l;
                    Point2f v2 = kp2r - kp2l;
                    double v1norm = sqrt( (pow(v1.x, 2) + pow(v1.y, 2)));
                    double v2norm = sqrt( (pow(v2.x, 2) + pow(v2.y, 2)));

                        if (v1norm  >= v2norm)
                        {
                            ofs<<(v1/v1norm).dot(v2/v1norm);
                            // cout<<calc[nibor[j]]<<calc[nibor[k]]<<" ";
                            // cout<< " v1 v2 " <<v1<<v2<< ""<<v1.dot(v2)<<endl;
                            // cout<< " vector dot :" <<(v1/v1norm).dot(v2/v1norm)<<endl;
                            // cout<< " cos :" <<(v1).dot(v2)/(v1norm*v2norm)<<endl;
                        }else {
                            // cout<<calc[nibor[j]]<<calc[nibor[k]]<<" ";
                            // cout<< " v1 v2 " <<v1<<v2<< ""<<v1.dot(v2)<<endl;
                            // cout<< " vector dot :" <<(v1/v2norm).dot(v2/v2norm)<<endl;
                            // cout<< " cos: " <<(v1).dot(v2)/(v1norm*v2norm)<<endl;
                            ofs<<(v1/v2norm).dot(v2/v2norm);
                        }
                                if (k != nibor.size()-1)
                                {
                                    ofs<<',';
                                }
                }
                // kp2l.x = kp2l.x / ori_img.cols;
                // kp2l.y = kp2l.y / ori_img.rows;
                // kp2r.x = kp2r.x / ori_img.cols;
                // kp2r.y = kp2r.y / ori_img.rows;
                ofs<<endl;



                //  在同一图像上的箭头 
                // if (calc[nibor[j]]==1)
                // {
                //     // for ( int  i = 0; i < final_match.size(); i++)
                //     // {
                //         circle(line_result,ori_kp[nibor[j]].pt, 1, cv::Scalar(0, 250, 0), 2);
                //         line(line_result,ori_kp[bf_matches[ nibor[j]].queryIdx].pt,
                //                          dst_kp[bf_matches[ nibor[j]].trainIdx].pt,cv::Scalar(255,0,0));
                //     // }

                // }
                

        }
        
      
    }

}

void wholegen(const vector<DMatch> bf_matches ,
            const vector<KeyPoint> ori_kp,
            const vector<KeyPoint> dst_kp,
            vector<int>  calc,
            Mat ori_img,
            Mat dst_img
           ){

    ofstream ofs;
    ofs.open("whole.csv", ios::out);



    vector<Point2f> ori_cord_list;
    for (const auto &m : bf_matches) {
        ori_cord_list.push_back(ori_kp[m.queryIdx].pt);
    }







    for (int i = 0 ; i<bf_matches.size() ; i++) 
    {




 vector<vector<double>> ori_dis_mat = distance_cal(ori_cord_list);
Point2f kp1l = ori_kp[bf_matches[i].queryIdx].pt;
        Point2f kp1r = dst_kp[bf_matches[i].trainIdx].pt;

        Point2f o1 = kp1l;
        Point2f o2 = kp1r;

        // kp1l = kp1l - Point2f(ori_img.cols/2,ori_img.rows/2);
        // kp1r = kp1r - Point2f(ori_img.cols/2,ori_img.rows/2);
        
        // kp1l.x = kp1l.x / ori_img.cols;
        // kp1l.y = kp1l.y / ori_img.rows;
        // kp1r.x = kp1r.x / ori_img.cols;
        // kp1r.y = kp1r.y / ori_img.rows;
                ofs << i <<','<< (kp1r.x -kp1l.x)/ ori_img.cols  
                         <<','<< (kp1r.y -kp1l.y)/ ori_img.rows
                         <<','<<kp1r.x/ ori_img.cols
                         <<','<<kp1l.x/ ori_img.cols
                         <<','<<kp1r.y/ ori_img.rows
                         <<','<<kp1l.y/ ori_img.rows
                         <<',';

        if(calc[i]==1) {ofs<<1<<','; }else {
    
            ofs <<0<<',';  
            // cout <<" 0:"<< o1<<o2 << kp1l << kp1r << endl;
            }


        for (int j= 0 ; j<bf_matches.size() ; j++){
                Point2f kp2l = ori_kp[bf_matches[j].queryIdx].pt;
                Point2f kp2r = dst_kp[bf_matches[j].trainIdx].pt;
                // kp2l = kp2l - Point2f(ori_img.cols/2,ori_img.rows/2);
                // kp2r = kp2r - Point2f(ori_img.cols/2,ori_img.rows/2);
            
                // kp2l.x = kp2l.x / ori_img.cols;
                // kp2l.y = kp2l.y / ori_img.rows;
                // kp2r.x = kp2r.x / ori_img.cols;
                // kp2r.y = kp2r.y / ori_img.rows;

                Point2f v1 = kp1r - kp1l;
                Point2f v2 = kp2r - kp2l;
                double v1norm = sqrt( (pow(v1.x, 2) + pow(v1.y, 2)));
                double v2norm = sqrt( (pow(v2.x, 2) + pow(v2.y, 2)));

                if (v1norm  > v2norm)
                {
                    ofs<<(v1/v1norm).dot(v2/v1norm);
                    // cout<<calc[i]<<calc[j]<<" ";
                    // cout<< " v1 v2 " <<v1<<v2<< ""<<v1.dot(v2)<<endl;
                    // cout<< " vector dot :" <<(v1/v1norm).dot(v2/v1norm)<<endl;
                    // cout<< " cos :" <<(v1).dot(v2)/(v1norm*v2norm)<<endl;
                }else {
                    // cout<<calc[i]<<calc[j]<<" ";

                    // cout<< " v1 v2 " <<v1<<v2<< ""<<v1.dot(v2)<<endl;
                    // cout<< " vector dot :" <<(v1/v2norm).dot(v2/v2norm)<<endl;
                    // cout<< " cos: " <<(v1).dot(v2)/(v1norm*v2norm)<<endl;
                    ofs<<(v1/v2norm).dot(v2/v2norm);
                }

                if (j != bf_matches.size()-1)
                {
                    ofs<<',';
                }
                // if (calc[i]==1&&calc[j]==1)
                // {
                        // for ( int  i = 0; i < final_match.size(); i++)
                        // {
                            // circle(line_result,ori_kp[final_match[i].queryIdx].pt, 1, cv::Scalar(0, 250, 0), 2);
                            // line(line_result,ori_kp[final_match[i].queryIdx].pt,dst_kp[final_match[i].trainIdx].pt,cv::Scalar(255,0,0));
                        // }

                // }
                

        }

        ofs <<endl;

        
      
    }

}







void csvgen(){
  string path = "./adelaidermf/";



  for (int i = 1; i < 39; i++)
  {


    vector<int> mask ;
    vector<DMatch> matches;

      string label = path + "label_" + to_string(i) + ".csv";

    std::ifstream file(label); 

    if (!file) {
        std::cout << "无法打开文件." << std::endl;
  
    }

    int lineCount = 0;
    std::string line;

    while (std::getline(file, line)) {
        if (stoi(line))mask.push_back(1);
        else mask.push_back(0);

        matches.push_back(DMatch(lineCount,lineCount,0));
        lineCount++;
        
    } // 匹配准备完毕
    cout<<"匹配准备完毕"<<lineCount<<endl;
    Mat img1 = imread(path+"img1_"+to_string(i)+".jpg");
    Mat img2 = imread(path+"img2_"+to_string(i)+".jpg");
imshow("",img1);
waitKey(0);
    vector<KeyPoint> kp1;
    vector<KeyPoint> kp2;



    // for (int i = 0; i < lineCount; i++)
    // {
         

            
                string data = path + "data_" + to_string(i) + ".csv";
            
                std::ifstream datafile(data); 
            
                if (!datafile) {
                    std::cout << "无法打开文件." << std::endl;
              
                }
     
                std::string ptline;
            
                while (std::getline(datafile, ptline)) {
                
                
                   std::vector<double> row; // 存储当前行的数字
                    
                    std::stringstream ss(ptline);
                    std::string cell;
                    
                    // 拆分当前行并将每个数字存储到vector中
                    while (std::getline(ss, cell, ',')) {
                        double number = std::stod(cell);
                        row.push_back(number);
                    }
                    kp1.push_back(KeyPoint(Point2f(row[0],row[1]),0));
                    kp2.push_back(KeyPoint(Point2f(row[3],row[4]),0));
                    // cout<<row[0]<<row[1]<<row[2]<<row[3] <<row[4]<<endl;           
                }

                  



    // }  // 特征点准备完毕
    cout<<"特征点准备完毕"<<endl;
    cout<<mask.size()<<endl;
    cout<<matches.size()<<endl;
    cout<<kp1.size()<<endl;
    cout<<kp2.size()<<endl;


vector<DMatch> rightmatch;
vector<DMatch> wrongmatch;
for (int i = 0; i < mask.size();i++){

  if (mask[i]==1 )
  {
    rightmatch.push_back(matches[i]);
    
  }else wrongmatch.push_back(matches[i]) ;
  
}

// cv::Mat matright, matwrong;
// drawMatches(img1,kp1,img2,kp2,rightmatch,matright);
// drawMatches(img1,kp1,img2,kp2,wrongmatch,matwrong);
// cout<<rightmatch.size()<<" " <<wrongmatch.size()<<endl;
// imshow("right",matright);
// imshow("wrong",matwrong);
// waitKey(0);










                string name = "./whole/from img1_+"+to_string(i)+
                                  +".csv";
                //  只能初始化const char×
                const char* newname = name.c_str();
                // datagen(matches,kp1,kp2,mask,img1,img2 );
                wholegen(matches,kp1,kp2,mask,img1,img2 );
                int result = std::rename("whole.csv",newname);


  }// 全部图像结束
  



}
Point2f rotatePoint(const Point2f& point, const Point2f& center, float angle) {
    float radians = (angle * static_cast<float>(CV_PI) / 180.0f);
    float cosTheta = std::cos(radians);
    float sinTheta = std::sin(radians);
    float x = point.x - center.x;
    float y = point.y - center.y;
    float xRot = x * cosTheta - y * sinTheta;
    float yRot = x * sinTheta + y * cosTheta;
    return Point2f(xRot + center.x, yRot + center.y);
}
//  new one  for using
int split(const string& name, int i, Mat& croppedImage1, Mat& croppedImage2, 
            int&  rotationType, Point2f& crop1Start, Point2f& crop2Start, float& rotationAngle, Size& size,
            int&  isresize,int& resizeType , float& resizeRatio

) {
    Mat originalImage = imread(name);
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distributionWidth(300, 800);
    std::uniform_int_distribution<int> distributionHeight(300, 800);
    std::uniform_real_distribution<float> distributionRotation(0.0f, 1.0f);
    std::uniform_real_distribution<float> distributionResize(0.0f, 1.0f);

    int cropWidth1 = distributionWidth(generator);
    int cropHeight1 = distributionHeight(generator);
    int x1 = std::uniform_int_distribution<int>(0, originalImage.cols - cropWidth1 - 1)(generator);
    int y1 = std::uniform_int_distribution<int>(0, originalImage.rows - cropHeight1 - 1)(generator);
    crop1Start = Point2f(x1, y1);


    // int cropWidth2 = cropWidth1;
    // int cropHeight2 = cropHeight1;


    int cropWidth2 = distributionWidth(generator);
    int cropHeight2 = distributionHeight(generator);
    int x2, y2;
    float intersectionThreshold = 0.10f;
    float intersectionArea = 0.0f;

    do {
        x2 = std::uniform_int_distribution<int>(std::max(0, x1 - cropWidth2), std::min(originalImage.cols - cropWidth2, x1 + cropWidth1))(generator);
        y2 = std::uniform_int_distribution<int>(std::max(0, y1 - cropHeight2), std::min(originalImage.rows - cropHeight2, y1 + cropHeight1))(generator);
        int intersectionX = std::max(x1, x2);
        int intersectionY = std::max(y1, y2);
        int intersectionWidth = std::min(x1 + cropWidth1, x2 + cropWidth2) - intersectionX;
        int intersectionHeight = std::min(y1 + cropHeight1, y2 + cropHeight2) - intersectionY;
        intersectionArea = intersectionWidth * intersectionHeight;
    } while (intersectionArea < intersectionThreshold * (cropWidth2 * cropHeight2));
    crop2Start = Point2f(x2, y2);

    cv::Rect cropRect1(x1, y1, cropWidth1, cropHeight1);
    croppedImage1 = originalImage(cropRect1).clone();

    cv::Rect cropRect2(x2, y2, cropWidth2, cropHeight2);
    croppedImage2 = originalImage(cropRect2).clone();





    float resizeThreshold = 0.5f;
    float randomResize = distributionResize(generator);
    resizeType = std::uniform_int_distribution<int>(0, 1)(generator);

    isresize = 0;
    if (randomResize<resizeThreshold)
    {
      resizeRatio = std::uniform_real_distribution<float>(0.4f, 1.0f)(generator);
      isresize = 1;
      if (resizeType == 0)
      {
        resize(croppedImage1,croppedImage1,Size(croppedImage1.cols*resizeRatio,croppedImage1.rows*resizeRatio));
      }else
      {

        resize(croppedImage2,croppedImage2,Size(croppedImage2.cols*resizeRatio,croppedImage2.rows*resizeRatio));

      }
      
    }
    










    float rotationThreshold = 0.5f;
    float randomRotation = distributionRotation(generator);
    rotationType = std::uniform_int_distribution<int>(0, 1)(generator);
    int rotated = 0;
    size = croppedImage1.size();

    if (randomRotation < rotationThreshold) {
        rotated = 1;
        float angle = std::uniform_real_distribution<float>(-120.0f, 120.0f)(generator);
        rotationAngle = angle;
        if (rotationType == 0) {
            size = croppedImage1.size();
            cv::Point2f center1(croppedImage1.cols / 2.0f, croppedImage1.rows / 2.0f);
            cv::Mat rotationMatrix1 = cv::getRotationMatrix2D(center1, angle, 1.0);
            cv::Rect bbox = cv::RotatedRect(center1, croppedImage1.size(), angle).boundingRect();
            rotationMatrix1.at<double>(0, 2) += bbox.width / 2.0 - center1.x;
            rotationMatrix1.at<double>(1, 2) += bbox.height / 2.0 - center1.y;
            cv::warpAffine(croppedImage1, croppedImage1, rotationMatrix1, bbox.size(), 1, 0, cv::Scalar(255, 255, 255));
        } else {
            size = croppedImage2.size();
            cv::Point2f center2(croppedImage2.cols / 2.0f, croppedImage2.rows / 2.0f);
            cv::Mat rotationMatrix2 = cv::getRotationMatrix2D(center2, angle, 1.0);
            cv::Rect bbox = cv::RotatedRect(center2, croppedImage2.size(), angle).boundingRect();
            rotationMatrix2.at<double>(0, 2) += bbox.width / 2.0 - center2.x;
            rotationMatrix2.at<double>(1, 2) += bbox.height / 2.0 - center2.y;
            cv::warpAffine(croppedImage2, croppedImage2, rotationMatrix2, bbox.size(), 1, 0, cv::Scalar(255, 255, 255));
        }
    } else {rotationAngle =0; rotationType = -1;}

    cv::imwrite("./out/" + to_string(i) + "l.png", croppedImage1);
    cv::imwrite("./out/" + to_string(i) + "r.png", croppedImage2);
    return rotated;
}



// final one
int all(int i,string name,vector<int>& calc ,Mat& croppedImage1,Mat& croppedImage2,
    std::vector<KeyPoint>& keypoints_1,std::vector<KeyPoint>& keypoints_2,
    std::vector<DMatch>& matches) {
    // Mat croppedImage1, croppedImage2;
    int rotationType;
    Point2f crop1Start, crop2Start;
    float rotationAngle;
    Size size;
int  isresize;int resizeType ; float resizeRatio;


int rotated = split(name, i, croppedImage1, croppedImage2, rotationType, crop1Start, crop2Start, rotationAngle, size,
                    isresize,resizeType,resizeRatio);



    float scale = 1.0;

    // cout <<"size"<<size<<endl;
    // cout <<"size"<<rotationType<<endl;
    // cout <<"size"<<croppedImage1.size().width<<endl;
    // cout <<"size"<<croppedImage2.size().width<<endl;




    // std::vector<KeyPoint> keypoints_1, keypoints_2;
    // std::vector<DMatch> matches;

    find_feature_matches(croppedImage1, croppedImage2, keypoints_1, keypoints_2, matches);
          vector<DMatch> rightmatch;
          vector<DMatch> wrongmatch;
    
    for (size_t i = 0; i < matches.size(); i++) {
        Point2f point1(keypoints_1[matches[i].queryIdx].pt);
        Point2f point2(keypoints_2[matches[i].trainIdx].pt);

        Point2f point1_orig, point2_orig;


        if (rotated != 0) {
          if (rotationType == 0) {
              scale = static_cast<float>(size.width) / static_cast<float>(croppedImage1.size().width);
              Point2f center1(croppedImage1.cols / 2.0f, croppedImage1.rows / 2.0f);
              Point2f rotatedPoint = rotatePoint(point1, center1, rotationAngle);
              rotatedPoint = rotatedPoint  -(center1- Point2f(size.width,size.height)/2) ;
              point1 = rotatedPoint;
            
            } else if (rotationType == 1) {
              scale = static_cast<float>(size.width) / static_cast<float>(croppedImage2.size().width);
              Point2f center2(croppedImage2.cols / 2.0f, croppedImage2.rows / 2.0f);
              Point2f rotatedPoint = rotatePoint(point2, center2, rotationAngle);
              rotatedPoint = rotatedPoint   - (center2- Point2f(size.width,size.height)/2);
              point2 = rotatedPoint;
            }
}



     




    if (isresize)
    {
      if (resizeType==0)
      {
        point1/=resizeRatio;
      }else
      {
        point2/=resizeRatio;
      }
      
      
    }
    

    // cout <<"scale"<<scale<<endl;
        point1_orig = point1 +crop1Start;
        point2_orig = point2 +crop2Start;









        // cout << point1_orig << " " << point2_orig << endl;
        // cout<<isresize<< rotated<<endl;
        double a =  euclideanDistance(point1_orig,point2_orig);
            if (a<7 )
            {
              rightmatch.push_back(matches[i]);
              calc.push_back(1);

            }else {
              wrongmatch.push_back(matches[i]) ;
              calc.push_back(0);
            }
    }

    






          // cv::Mat matright, matwrong;
          // drawMatches(croppedImage1,keypoints_1,croppedImage2,keypoints_2,rightmatch,matright);
          // drawMatches(croppedImage1,keypoints_1,croppedImage2,keypoints_2,wrongmatch,matwrong);
          // cout<<rightmatch.size()<<" " <<wrongmatch.size()<<endl;
          // imshow("right",matright);
          // imshow("wrong",matwrong);
          // waitKey(0);
          // destroyAllWindows();






    return std::count(calc.begin(),calc.end(),1);
}


int main(int argc , char** argv){

// string Camerapath = argv[1];
// string Fpath = argv[2];
// // count lines

//     std::ifstream file(Fpath);  

//     if (!file) {
//         std::cout << "无法打开文件." << std::endl;
//         return 1;
//     }

//     int lineCount = 0;
//     std::string line;

//     while (std::getline(file, line)) {
//         lineCount++;
//     }

//     file.close();




// for (int i = 0; i <lineCount; i++)
// {
//     single(Camerapath,Fpath ,i);
    
// }



// csvgen();




    std::string folderPath = argv[1]; 
    
    std::vector<std::string> fileNames = readFolder(folderPath);

    for (int i=0; i<3000; i++) {
        // for (int j=0; j<fileNames.size(); j+=7) {
          // if (i == j) continue;
  
          Mat img1 ,img2;
          // split(argv[1],i,img1,img2);

          // img1 = imread(folderPath+fileNames[i]);
          // img2 = imread(folderPath+fileNames[j]);
          // imshow("",img1);
          // waitKey(0);
          

          vector<KeyPoint> kp1 ,kp2;
          vector<DMatch> matches;
          // find_feature_matches(img1,img2,kp1,kp2,matches);
          // vector<int> mask;
          // int k = pose_estimation_2d2d(kp1,kp2,matches,mask);

          vector<int> calc;
          int right = all(i,argv[1],calc,img1,img2,kp1,kp2,matches) ;

          double inlinerrate = static_cast<double> (right)/ static_cast<double>(matches.size());
          cout<<inlinerrate<<endl;
          if (right<30||inlinerrate>0.2) continue;
          // int right = 0;

          // for (int l = 0; l < mask.size(); l++) 
          // {
          //   if((mask[l])==1){ calc.push_back(1); right ++;} else calc.push_back(0);
          // }
          cout<<right<<" "<<matches.size()<<" "<<kp1.size()<<endl;
          datagen(matches,kp1,kp2,calc,img1,img2);
          





          string name = "./last/from"
                                      // +fileNames[i]
                                    // +fileNames[j]
                                    +to_string(i)+"_"
                                    +to_string(right)
                                    +".csv";
          //  只能初始化const char×
          const char* newname = name.c_str();
          // datagen(matches,kp1,kp2,mask,img1,img2 );

          int result = std::rename("test.csv",newname);








          // vector<DMatch> rightmatch;
          // vector<DMatch> wrongmatch;
          // for (int a = 0; a < calc.size();a++){
          
          //   if (calc[a]==1 )
          //   {
          //     rightmatch.push_back(matches[a]);

          //   }else wrongmatch.push_back(matches[a]) ;

          // }

          // cv::Mat matright, matwrong;
          // drawMatches(img1,kp1,img2,kp2,rightmatch,matright);
          // drawMatches(img1,kp1,img2,kp2,wrongmatch,matwrong);
          // cout<<rightmatch.size()<<" " <<wrongmatch.size()<<endl;
          // imshow(to_string(i),matright);
          // imshow("wrong",matwrong);
          // waitKey(0);




        // }
        

    }



























return 0;

}