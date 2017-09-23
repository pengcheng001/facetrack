#ifndef RECOGNITION_H
#define RECOGNITION_H

#include<stdio.h>
#include<iostream>
#include<vector>
#include<string>

#include<opencv2/opencv.hpp>
#include<opencv2/face.hpp>

#include"define.h"



using namespace std;
using namespace cv;
using namespace cv::face;
enum FACE_REC_METHOD{ FISHERFACE=0,EIGENFACES};

class recognitionFace{
    private:
        cv::Ptr<cv::face::FaceRecognizer>  model;
        bool haveModel ;
    public:
        cv::Ptr<cv::face::FaceRecognizer> learnCollectedFacesAndTrain( const std::vector<Mat> &preprocFaces,  const std::vector<int> &labels,  const FACE_REC_METHOD facerecAlgorithm);
        bool predict(INPUT cv::Mat *face ,OUTPUT  int &predictLabel, OUTPUT double &confidence);
        recognitionFace();
         bool   isHaveModel(){
            return haveModel;
        }
         cv::Ptr<cv::face::FaceRecognizer>  getModel();

};
//not used  for below ,just for debug
cv::Ptr<cv::face::FaceRecognizer>    learnCollectedFacesAndTrain(const std::vector<cv::Mat> &preprocFaces,
                                                      const std::vector<int> &labels,
                                                      const FACE_REC_METHOD facerecAlgorithm);
double getSimilaryity(const cv::Mat A, const cv::Mat B);
bool preditFromModel(cv::Ptr<cv::face::FaceRecognizer>  model);
cv::Mat reconstructFace(const cv::Ptr<cv::face::FaceRecognizer> model, const cv::Mat perprocessedFace);
#endif // RECOGNITION_H
