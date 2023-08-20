#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <string>

using namespace std;
using namespace cv;

float euclideanDistance(const Point2f& p1, const Point2f& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return sqrt(dx*dx + dy*dy);
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


int split(const string& name, int i, Mat& croppedImage1, Mat& croppedImage2, 
int& rotationType, Point2f& crop1Start, Point2f& crop2Start, float& rotationAngle, Size& size) {
    Mat originalImage = imread(name);
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distributionWidth(300, 1200);
    std::uniform_int_distribution<int> distributionHeight(300, 1200);
    std::uniform_real_distribution<float> distributionRotation(0.0f, 1.0f);

    int cropWidth1 = distributionWidth(generator);
    int cropHeight1 = distributionHeight(generator);
    int x1 = std::uniform_int_distribution<int>(0, originalImage.cols - cropWidth1 - 1)(generator);
    int y1 = std::uniform_int_distribution<int>(0, originalImage.rows - cropHeight1 - 1)(generator);
    crop1Start = Point2f(x1, y1);

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

    float rotationThreshold = 0.9f;
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

    cv::imwrite(to_string(i) + "l.png", croppedImage1);
    cv::imwrite(to_string(i) + "r.png", croppedImage2);
    return rotated;
}






void find_feature_matches(const Mat& img_1, const Mat& img_2,
                          std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2,
                          std::vector<DMatch>& matches) {
    Ptr<SIFT> sift = SIFT::create(3000);

    Mat descriptors_1, descriptors_2;

    sift->detectAndCompute(img_1, noArray(), keypoints_1, descriptors_1);
    sift->detectAndCompute(img_2, noArray(), keypoints_2, descriptors_2);

    Ptr<BFMatcher> bf = BFMatcher::create(NORM_L2, true);
    bf->match(descriptors_1, descriptors_2, matches);
}

int main(int argc, char** argv) {
    Mat croppedImage1, croppedImage2;
    int rotationType;
    Point2f crop1Start, crop2Start;
    float rotationAngle;
    Size size;
int rotated = split(argv[1], 10, croppedImage1, croppedImage2, rotationType, crop1Start, crop2Start, rotationAngle, size);



    float scale = 1.0;

    cout <<"size"<<size<<endl;
    cout <<"size"<<rotationType<<endl;
    cout <<"size"<<croppedImage1.size().width<<endl;
    cout <<"size"<<croppedImage2.size().width<<endl;




    std::vector<KeyPoint> keypoints_1, keypoints_2;
    std::vector<DMatch> matches;

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


    cout <<"scale"<<scale<<endl;
        point1_orig = point1 +crop1Start;
        point2_orig = point2 +crop2Start;
     








        cout << point1_orig << " " << point2_orig << endl;

        double a =  euclideanDistance(point1_orig,point2_orig);
            if (a<5 )
            {
              rightmatch.push_back(matches[i]);

            }else wrongmatch.push_back(matches[i]) ;
    }

    
          cv::Mat matright, matwrong;
          drawMatches(croppedImage1,keypoints_1,croppedImage2,keypoints_2,rightmatch,matright);
          drawMatches(croppedImage1,keypoints_1,croppedImage2,keypoints_2,wrongmatch,matwrong);
          cout<<rightmatch.size()<<" " <<wrongmatch.size()<<endl;
          imshow("right",matright);
          imshow("wrong",matwrong);
          waitKey(0);






    return 0;
}
