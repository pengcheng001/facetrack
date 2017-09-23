
//#include <QCoreApplication>
#include<opencv2/opencv.hpp>
#include<opencv2/face.hpp>


#include"Utils.hpp"
#include"detectFace.hpp"
#include"preprocessface.hpp"
using namespace cv;

int main(int argc, char *argv[])
{
//    QCoreApplication a(argc, argv);
    int camerNum =0;
    if(argc > 1)  camerNum = atoi(argv[1]);
    cv::VideoCapture cap;
    getCamera(cap,camerNum);
    processVideo(cap);

    return 0;

//    return a.exec();
}
