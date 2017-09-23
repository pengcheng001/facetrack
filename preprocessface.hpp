#ifndef PREPROCESSFACE_H
#define PREPROCESSFACE_H


#include <opencv2/opencv.hpp>
#include<vector>


class preprocessFace
{
    private:
        cv::Mat face;
        cv::Mat warpedMat;
        cv::Point2f leftPoint;
        cv::Point2f rightPoint;
        bool isGetFace;
    public:
        preprocessFace();
        void initFace(const std::vector<cv::Rect> *faceInfo, const cv::Mat *frame);
        bool isHaveFace();
        void warpingFace();
        void eqHist();
        void CreateMast();
        cv::Mat getProcessedFace();
        void clean();

};

#endif // PREPROCESSFACE_H
