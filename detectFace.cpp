
#include"detectFace.hpp"

using namespace cv;

static std::string leftEyeXmlPath = "";
const int scaleWeidth = 300;

void trans2gray(const cv::Mat *src,cv::Mat &dest){
    if(3 == src->channels())
    {
        cv::cvtColor(*src,dest,cv::COLOR_BGR2GRAY);
    }else if(4 == src->channels())
    {
        cv::cvtColor(*src,dest,cv::COLOR_BGRA2GRAY);
    }else{
        dest = *src;
    }
}

void resizeInputImage(const cv::Mat *src, cv::Mat &dest, int scalewidth){
    if( src->cols > scalewidth)
    {
        float scale = src->cols/(float)scalewidth;
        int scaleHeight = cvRound((src->rows/scale));
        cv::resize(*src,dest,cv::Size(scalewidth,scaleHeight));
    }else{
        dest = *src;
    }
}


void adjustImage(const cv::Mat *src, cv::Mat &dest,int scaleWidth){
    trans2gray(src,dest);
    resizeInputImage(&dest,dest,scaleWidth);
   // cv::Scalar  avg =cv::mean(dest);
    //dest = dest-avg;

    cv::equalizeHist(dest,dest);
    //cv::imshow("t",dest);
}

void rAdjustImage(cv::Rect &rect,float scale)
{

    rect.x = (int)(rect.x *scale);
    rect.y = (int)(rect.y *scale);
    rect.width = (int)(rect.width *scale);
    rect.height = (int)(rect.height *scale);

    if(rect.size().area() <= 0)
    {
        rect = cv::Rect(-1,-1,-1,-1);
    }
}


void detectFaces(cv::CascadeClassifier *ccf, const cv::Mat *input, cv::Rect &face)
{
    cv::Mat scanImge;
    std::vector<cv::Rect> faces;
    float scale = input->cols /(float)scaleWeidth;
    adjustImage(input,scanImge,scaleWeidth);
    int detectModel = cv::CASCADE_FIND_BIGGEST_OBJECT||0;
    cv::Size minSize(30,30);
    detectObject(ccf,&scanImge,faces,detectModel,&minSize);
    if(faces.size() > 0)
    {
        face = faces[0];
        if(input->cols > scaleWeidth) rAdjustImage(face,scale);
    }
    else{
        face = cv::Rect(-1,-1,-1,-1);
    }
}

void detectEye(cv::CascadeClassifier *ccfEye,cv::Mat *input,cv::Rect &eye){
    cv::Mat scanMat;

    std::vector<cv::Rect> rects;
    trans2gray(input,scanMat);
    cv::equalizeHist(scanMat,scanMat);
    cv::Size minSize(5,5);
    rects.clear();
    int detectModel = cv::CASCADE_FIND_BIGGEST_OBJECT||0;
    detectObject(ccfEye,&scanMat,rects,detectModel,&minSize);
    if(rects.size() >0 && rects[0].x != 0 && rects[0].y != 0)
        eye = rects[0];
    else eye = cv::Rect(-1,-1,-1,-1);
}

void detectObject(cv::CascadeClassifier *cascade,const cv::Mat *frame, std::vector<cv::Rect> &rects, int flage, cv::Size *minSize)
{
    //::cout<<(*frame)<<std::endl;
    if(frame->cols >minSize->width && frame->rows>minSize->height)
        try{
       // std::cout<<frame->channels()<<std::endl;
        cascade->detectMultiScale(*frame,rects,1.1,4,flage,*minSize);
    }catch(cv::Exception e){
        std::cout<<e.code<<std::endl;
    }
    else std::cerr<<"Could not detect"<<std::endl;

}


void detectFace(cv::CascadeClassifier *faceDetector,cv::CascadeClassifier *leftDetector,cv::CascadeClassifier *rightDetector,
                cv::Mat *frame, std::vector<cv::Rect> &face)
{
    cv::Rect facerect;
    cv::Mat faceMat;
    cv::Mat leftFace;
    cv::Mat rightFace;
    cv::Rect eye;


    face.clear();
    detectFaces(faceDetector,frame,facerect);

    if(facerect.x != -1 && facerect.y != -1)
    {
        face.push_back(facerect);
        //faceMat = (*frame)(facerect);
        (*frame)(facerect).copyTo(faceMat);
        leftFace = faceMat(cv::Range::all(),cv::Range(1,faceMat.cols/2));
        rightFace = faceMat(cv::Range::all(),cv::Range(faceMat.cols/2,faceMat.cols));

     //   detectEye(leftDetector,&leftFace,0||cv::CASCADE_FIND_BIGGEST_OBJECT,eye);
        detectEye(leftDetector,&leftFace,eye);

        if(eye.x != -1 && eye.y != -1){
            face.push_back(eye);
            detectEye(rightDetector,&rightFace,eye);
            if(eye.x != -1 && eye.y != -1)
            {

                //std::cout<<"x:"<<eye.x<<" y:"<<eye.y<<" width:"<<eye.width<<" height:"<<eye.height<<std::endl;
               // std::cout<<"x:"<<face[1].x<<" y:"<<face[1].y<<" width:"<<face[1].width<<" height:"<<face[1].height<<std::endl;
                eye.x += faceMat.cols/2;
                face.push_back(eye);
                cv::rectangle(faceMat,eye,cv::Scalar(0,255,0),2);
                cv::rectangle(faceMat,face[1],cv::Scalar(0,255,0),2);
               // cv::imshow("left",leftFace);
               // cv::imshow("right",rightFace);
                cv::imshow("face",faceMat);

            }else{
                face.clear();
            }
        }


      //  std::cout<<eyes.size()<<std::endl;
    }

}


void detectAndDisplay( Mat &frame , cv::CascadeClassifier *face_cascade)
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade->detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
 /*
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        Mat faceROI = frame_gray( faces[i] );
        std::vector<Rect> eyes;

        //-- In each face, detect eyes

        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );

        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }

    }*/
    //-- Show what you got
    imshow( "window_name", frame );
}
