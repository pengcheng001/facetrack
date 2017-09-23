#ifndef UTILS_HPP
#define UTILS_HPP

#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include"detectFace.hpp"
#include"preprocessface.hpp"
#include"Recognition.hpp"
#include"define.h"

using namespace std;



cv::Rect drawText(INANDOUT cv::Mat &img, INPUT string text, INPUT cv::Point coord, INPUT  cv::Scalar color, INPUT  float fontScale = 0.5f,  INPUT int thicness = 2, INPUT int fontFace = cv::FONT_HERSHEY_SIMPLEX);
cv::Rect drawButton(INANDOUT cv::Mat &img, INPUT  string text,  INPUT cv::Point coord, INPUT int minWidth =0);
bool isButtonRect(const cv::Point pt, const cv::Rect rect);
void getCamera(cv::VideoCapture &camera, int cameraNum);
void initButtons(cv::Mat &img);
void createFacedetetor(cv::CascadeClassifier &faceDetetor, const std::string xmlPath);

void onMouse(int event, int x ,int y, int , void *);

void whiteBalance(const cv::Mat *src, cv::Mat dest);

void processVideo(cv::VideoCapture &capture);
#endif // UTILS_HPP
