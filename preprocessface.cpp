#include "preprocessface.hpp"


const double FACEWIEDTHTATE =1.9;
const double DESIRED_LEFT_EYE_X= 0.16;
const double DESIRED_RIGHT_EYE_Y =(1.0f - 0.16);

const int DESIRED_FACE_WIDTH = 70;
const int DESIRED_FACE_HEIGHT = 70;
preprocessFace::preprocessFace():isGetFace(false)
{

}

void preprocessFace::initFace(const std::vector<cv::Rect> *faceInfo, const cv::Mat *frame)
{
    if(faceInfo->size() == 3 && !frame->empty())
    {
        //this->face = frame->clone()((*faceInfo)[0]);
        frame->copyTo(face);
        face= face((*faceInfo)[0]);
        leftPoint = cv::Point2f((*faceInfo)[1].x + (*faceInfo)[1].width/2.0f,(*faceInfo)[1].y + (*faceInfo)[1].height*(2/3.0f));
        rightPoint = cv::Point2f((*faceInfo)[2].x + (*faceInfo)[2].width/2.0f,(*faceInfo)[2].y + (*faceInfo)[2].height*(2/3.0f));
        if (cv::abs(rightPoint.y - leftPoint.y) < face.rows/3.0) isGetFace = true;
       // cv::circle(face,leftPoint,5,cv::Scalar(255,0,0),2);
        //cv::circle(face,rightPoint,5,cv::Scalar(255,0,0),2);

  //      cv::imshow("faceImg",face);
    }

}

bool preprocessFace::isHaveFace()
{
    return isGetFace;
}

void preprocessFace::warpingFace()
{
    if (isGetFace)
    {
        cv::Point2f eyesCenter;
        cv::Mat filterMat;
        eyesCenter.x = (leftPoint.x + rightPoint.x)*0.5f;
        eyesCenter.y = (leftPoint.y + rightPoint.y)*0.5f;

        double dy = (rightPoint.y - leftPoint.y);
        double dx = (rightPoint.x - leftPoint.x);
        double len = std::sqrt(dx*dx + dy*dy)*1.2;
        double angle = std::atan2(dy,dx) *180/CV_PI;

       double desirLen = (DESIRED_RIGHT_EYE_Y - 0.16);
       double scale = desirLen * DESIRED_FACE_WIDTH/len;

       cv::Mat rot_mat = cv::getRotationMatrix2D(eyesCenter,angle,scale);

       double ex = DESIRED_FACE_WIDTH * 0.5f - eyesCenter.x;
       double ey = DESIRED_FACE_HEIGHT * DESIRED_RIGHT_EYE_Y - eyesCenter.y;

       rot_mat.at<double>(0,2) += ex;
       rot_mat.at<double>(1,2) += ey;
       warpedMat = cv::Mat(DESIRED_FACE_WIDTH*FACEWIEDTHTATE,DESIRED_FACE_HEIGHT,CV_8U,cv::Scalar(128));
       //imshow("face11",face);
       cv::Mat gray;
       cv::cvtColor(face,gray,cv::COLOR_BGR2GRAY);
      // imshow("face11",gray);
       cv::warpAffine(gray,warpedMat,rot_mat,warpedMat.size());
      // imshow("warped",warpedMat);
       eqHist();
       filterMat = cv::Mat(warpedMat.size(),CV_8U);
       cv::bilateralFilter(warpedMat,filterMat,0,15.0,15.0);
       warpedMat = filterMat;
      // cv::imshow("filterMat",warpedMat);
       CreateMast();

    }else {
        warpedMat = cv::Mat();
    }
}

void preprocessFace::eqHist()
{
    if (isHaveFace())
    {
        int w = warpedMat.cols;
        int h = warpedMat.rows;
        cv::Mat wholeFace;
        cv::Mat leftFace;
        cv::Mat rightFace;
        int mid_x = w/2;

        cv::equalizeHist(warpedMat, wholeFace);
        leftFace = warpedMat(cv::Rect(0,0,mid_x,h));
        rightFace =warpedMat(cv::Rect(mid_x,0,w-mid_x,h));
        cv::equalizeHist(leftFace,leftFace);
        cv::equalizeHist(rightFace,rightFace);

        for(int y = 0; y < h; y++)
        {
            for(int x =0; x < w; x++){
                int v;
                if(x < w/4) v = leftFace.at<uchar>(y,x);
                else if( x < w/2)
                {
                    int lv =leftFace.at<uchar>(y,x);
                    int wv = wholeFace.at<uchar>(y,x);
                    float rate = (x - w/4) / (float)(w/4);
                    v = cvRound((1.0f - rate) * lv + rate * wv);
                }else if( x < w*3/4)
                {
                    int rv = rightFace.at<uchar>(y,x-mid_x);
                    int wv = wholeFace.at<uchar>(y,x);

                    float rate = (x - w/2) / (float)(w/4);
                    v = cvRound((1.0f - rate) * wv + rate * rv);
                }else
                {
                    v = rightFace.at<uchar>(y,x-mid_x);
                }
                warpedMat.at<uchar>(y,x) = v;
            }
        }
      //  std::cout<<warpedMat<<std::endl;
    }
   // cv::imshow("hist",warpedMat);
}

void preprocessFace::CreateMast()
{
    //cv::Mat mask = cv::Mat(warpedMat.size(),CV_8SC1,cv::Scalar(255));
    cv::Mat mask = cv::Mat::zeros(warpedMat.size(),CV_8SC1);

    double dw = warpedMat.cols;
    double dh = warpedMat.rows;
    cv::Point faceCenter = cv::Point(cvRound(dw*0.5),
                                     cvRound(dh)*0.6);
    cv::Size size = cv::Size(cvRound(dw*0.5), cvRound(dh*0.5));
    cv::ellipse(mask,faceCenter,size,0,0,360,cv::Scalar(1),-1);

    for(int x = 0; x > dw; x++)
    {
        for(int y= 0; y > dh; y++)
        {
            if(mask.at<uchar>(x,y) == 0)
                warpedMat.at<uchar>(x,y) = 0;
        }
    }
   // std::cout<<"mask==>cols:"<<mask.cols<<" rows:"<<mask.rows<<"channel:"<<mask.channels()<<std::endl;
   // std::cout<<"face==>cols:"<<warpedMat.cols<<" rows:"<<warpedMat.rows<<"channel:"<<warpedMat.channels()<<std::endl;

    cv::imshow("mask",warpedMat);
}

cv::Mat preprocessFace::getProcessedFace()
{
    if(isGetFace)
    {
        warpingFace();
        return this->warpedMat;
    }else {
        return cv::Mat();
    }
}

void preprocessFace::clean()
{
    isGetFace = false;
    face.release();
    warpedMat.release();
}
