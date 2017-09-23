#ifndef DETECTFACE_HPP
#define DETECTFACE_HPP

#include<stdio.h>
#include<iostream>
#include<vector>

#include<opencv2/opencv.hpp>
using namespace cv;
//detect face from image
void detectObject(cv::CascadeClassifier *cascade,const cv::Mat *frame, std::vector<cv::Rect> &rects, int flage, cv::Size *minSize);
void detectFace(cv::CascadeClassifier *faceDetector,cv::CascadeClassifier *leftDetector,
                cv::CascadeClassifier *rightDetector,
                cv::Mat *frame, std::vector<cv::Rect> &face);


void detectAndDisplay( Mat &frame , cv::CascadeClassifier *face_cascade);
void detectFaces(cv::CascadeClassifier *ccf, const cv::Mat *input, cv::Rect &face);


#endif // DETECTFACE_HPP
