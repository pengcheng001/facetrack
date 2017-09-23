
#include"Utils.hpp"


const int WINDOW_CAMERA_WIDTH =  640;
const int WINDOW_CAMERA_HEIGHT =  640;
const cv::Point FRIST_BUTTON_POINT = cv::Point(5,25);
const int GAP = 5;
const double FACE_DIFF = 0.3;
const double TIME_DIFF = 1.0;

FACE_REC_METHOD recMethod = FISHERFACE;

int label = -1;
int countFaces = 0;
int personsNum = 0;
cv::Rect m_rcBtnAdd;
cv::Rect m_rcBtnDel;
cv::Rect m_rcBtnTraing;
cv::Rect m_rcBtnRecogtion;
cv::Rect m_rcBtnDetect;
cv::Rect m_rcBtnExit;


enum MODES {MODE_STARTUP =0, MODE_DETECTON, MODE_COLLECT_FACE, MODE_TRANING, MODE_RECONGNITION,MODE_DELETE_ALL, MODE_END};
const string MODE_NAMES[] = {"Startup","Detection","Collect Faces","Traing","Recogtion","Delete All","Error"};
MODES m_mode = MODE_DETECTON;


std::string faceXmlPath ="/home/pch/opencv/opencv-3.0.0-beta/data/haarcascades/haarcascade_frontalface_alt2.xml";
//std::string leftEyePath = "/home/pch/haarcascades/haarcascade_lefteye_2splits.xml";
//std::string leftEyePath = "/home/pch/haarcascades/haarcascade_mcs_eyepair_big.xml";
std::string leftEyePath = "/home/pch/opencv/opencv-3.0.0-beta/data/haarcascades/haarcascade_eye.xml";
std::string rightEyePath = "/home/pch/opencv/opencv-3.0.0-beta/data/haarcascades/haarcascade_eye.xml";
//std::string rightEyePath = "/home/pch/haarcascades/haarcascade_righteye_2splits.xml";
//std::string rightEyePath = "/home/pch/haarcascades/haarcascade_mcs_eyepair_big.xml";
//std::string rightEyePath = "/home/pch/opencv-3.0.0-beta/data/haarcascades/haarcascade_righteye_2splits.xml";


cv::CascadeClassifier ccfFace;
cv::CascadeClassifier ccleftfEye;
cv::CascadeClassifier ccfrightEye;


cv::Mat frame;
cv::Rect face_t;
std::vector<cv::Rect> faces;


void initCascade(){
    createFacedetetor(ccfFace,faceXmlPath);
    createFacedetetor(ccleftfEye,leftEyePath);
    createFacedetetor(ccfrightEye,rightEyePath);

}

double getSimilarity(const cv::Mat &A, const cv::Mat &B)
{
    double errorL2 = cv::norm(A,B,cv::NORM_L2);

    double similarity = errorL2 /(double)(A.rows*A.cols);

    return similarity;
}

void processVideo(cv::VideoCapture &capture)
{
    cv::Mat frame;
    initCascade();
    preprocessFace prf;
    cv::namedWindow("display",cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("display",onMouse,0);
    double old_time = (double)cv::getTickCount();
    cv::Mat old_face;
    std::vector<cv::Mat> *preprocessFaces = new std::vector<cv::Mat>();
    std::vector<int> *labels = new std::vector<int>();
    recognitionFace *recFaceModel = new recognitionFace();

    while(true)
    {
        capture >> frame;
        if(!frame.data)
        {
            std::cout<<"Error: Could not get image data"<<std::endl;
            continue;
        }
        if(m_mode == MODE_DETECTON)
        {
            detectFace(&ccfFace,&ccleftfEye,&ccfrightEye,&frame,faces);
            //prf.initFace(&faces,&frame);
           // prf.warpingFace();
            if(faces.size()>0)
            {
                face_t = faces[0];
                cv::rectangle(frame,face_t,cv::Scalar(0,255,0),2);
            }

        }else if (m_mode == MODE_COLLECT_FACE) {
            double current_time = old_time;
            detectFace(&ccfFace,&ccleftfEye,&ccfrightEye,&frame,faces);
            prf.initFace(&faces,&frame);
            if(faces.size()>= 3)
                current_time = (double)cv::getTickCount();
            if(!old_face.data)
                old_face = prf.getProcessedFace();
            else{
                double timeDiff_second = (current_time - old_time)/cv::getTickFrequency();
                cv::Mat current_face ;//= prf.getProcessedFace().clone();
                prf.getProcessedFace().copyTo(current_face);
                double imageDiff = getSimilarity(old_face,current_face);

                if(timeDiff_second > TIME_DIFF && imageDiff > FACE_DIFF)
                {
                    cv::Mat mirroredFace;
                    cv::flip(current_face,mirroredFace,1);
                    preprocessFaces->push_back(current_face);
                    preprocessFaces->push_back(mirroredFace);
                    labels->push_back(label);
                    labels->push_back(label);
                    old_face = current_face;
                    old_time = current_time;
                    cv::Mat displayImg = frame(faces[0]);
                    displayImg += cv::Scalar(90,90,90);
                    //
                    std::cout<<++countFaces<<" faces to add for traning"<<std::endl;
                }

            }
            if(faces.size()>0)
            {
                face_t = faces[0];
                cv::rectangle(frame,face_t,cv::Scalar(0,255,0),2);
            }

        }else if ( m_mode == MODE_TRANING) {
            bool isHaveEnoughFaces = true;
            if(recMethod == FISHERFACE){
                if(personsNum < 2)
                {
                    std::cout<<"fisherface need least 2 person faces,there is no enough"<<std::endl;
                    isHaveEnoughFaces = false;
                }

            }
            if(personsNum < 1)
            {
                std::cout<<"need more persons faces"<<std::endl;
                isHaveEnoughFaces = false;
            }
            if(isHaveEnoughFaces)
            {
             //   model = learnCollectedFacesAndTrain(*preprocessFaces,*labels,recMethod);
                recFaceModel->learnCollectedFacesAndTrain(*preprocessFaces,*labels,recMethod);
                m_mode = MODE_RECONGNITION;
                
            }
            else{
                m_mode = MODE_COLLECT_FACE;
            }
        }else if ( m_mode == MODE_RECONGNITION) {
           detectFace(&ccfFace,&ccleftfEye,&ccfrightEye,&frame,faces);
           prf.initFace(&faces,&frame);
           cv::Mat preprocessFace = prf.getProcessedFace();
           prf.clean();
            if(preprocessFace.data)
            {
                int label = -1;
                double confidence = 0.0;
                std::stringstream resultMessage;
                if(recFaceModel->predict(&preprocessFace,label,confidence))
                {
                    if(label == -1)
                        resultMessage<<"unKonwPerson"<<std::endl;
                    else
                        resultMessage<<"predict label is:"<<label<<" , Confidence is:"<<confidence<<endl;
                }
                else {
                    resultMessage<<"ERROR: No model had been trained for predict"<<std::endl;
                }
                std::cout<<resultMessage.str();
                drawText(frame,resultMessage.str(),cv::Point(faces[0].x , faces[0].y),cv::Scalar(255,0,0));
            }
        }
        else if (m_mode == MODE_END) {
            break;
        }
         initButtons(frame);
        cv::imshow("display",frame);
        cv::waitKey(10);

    }
    delete preprocessFaces;
    delete labels;
}


void onMouse(int event, int x, int y, int , void *)
{
    if(cv::EVENT_LBUTTONDOWN != event)
        return;
    cv::Point point = cv::Point(x,y);
    if(isButtonRect(point,m_rcBtnAdd))
    {
        std::cout<<"add one person for traing"<<std::endl;
        std::cout<<"please input a label(nunber):";
        std::cin>>label;
       personsNum++;
        m_mode = MODE_COLLECT_FACE;
    }else if (isButtonRect(point,m_rcBtnDel)) {
        std::cout<<"delete all faces"<<std::endl;
        m_mode = MODE_DELETE_ALL;
    }else if (isButtonRect(point,m_rcBtnDetect)) {
        std::cout<<"detect face"<<std::endl;
        m_mode = MODE_DETECTON;
    }else if(isButtonRect(point,m_rcBtnRecogtion))
    {
        std::cout<<"recogtion face"<<std::endl;
        m_mode = MODE_RECONGNITION;
    }else if(isButtonRect(point,m_rcBtnTraing))
    {
        std::cout<<"traning..."<<std::endl;
        m_mode = MODE_TRANING;
    }else if(isButtonRect(point,m_rcBtnExit))
    {
        std::cout<<"exit......"<<std::endl;
        personsNum++;
        m_mode = MODE_END;
    }
}
void initButtons(cv::Mat &img)
{
    cv::Point point = FRIST_BUTTON_POINT;
    m_rcBtnDetect = drawButton(img,"Detect Face",point);
    point.y = m_rcBtnDetect.y + m_rcBtnDetect.height + GAP;
    m_rcBtnAdd = drawButton(img,"Add Faces",point);
    point.y = m_rcBtnAdd.y + m_rcBtnAdd.height + GAP;
    m_rcBtnTraing = drawButton(img,"Traing",point);
    point.y = m_rcBtnTraing.y + m_rcBtnTraing.height + GAP;
    m_rcBtnRecogtion = drawButton(img,"Recogtion",point);
    point.y = m_rcBtnRecogtion.y + m_rcBtnRecogtion.height + GAP;
    m_rcBtnDel = drawButton(img,"Delete ALL",point);
    point.y = m_rcBtnDel.y + m_rcBtnDel.height + GAP;
    m_rcBtnExit = drawButton(img,"EXIT",point);

}
void getCamera(cv::VideoCapture &camera, int cameraNum){
    int cmrNum =  cameraNum;
    camera.open(cmrNum);
    if(!camera.isOpened())
    {
        std::cerr<<"Error: Counld not access to the camera or video("<<cmrNum<<")"<<std::endl;
        exit(0);
    }else{
        std::cout<<"camer: "<<cmrNum<<" open successfuly!"<<std::endl;
    }

    camera.set(cv::CAP_PROP_FRAME_WIDTH,WINDOW_CAMERA_WIDTH);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT,WINDOW_CAMERA_HEIGHT);
}


void createFacedetetor(cv::CascadeClassifier &faceDetetor, const std::string xmlPath){

    try{
        if(!faceDetetor.load(xmlPath))
        {
            std::cerr<<"ERROR: Could not load Face Detetor"<<std::endl;
            exit(1);
        }else{
            std::cout<<xmlPath<<" is loaded sucessfuly!!"<<std::endl;
        }
    }catch(cv::Exception e){
        std::cerr<<"ERROR: Could not load Face Detetor"<<std::endl;
        exit(1);
    }

}

void whiteBalance(const cv::Mat *src, cv::Mat dest)
{
    std::vector<cv::Mat> vchannels;

    cv::split(*src,vchannels);

    cv::Mat BlueChannel = vchannels[0];
    cv::Mat GreenChannel = vchannels[1];
    cv::Mat RedChannel = vchannels[2];

    double blueAvg = 0;
    double greenAvg =0;
    double redAvg =0;

    blueAvg = cv::mean(BlueChannel)[0];
    greenAvg = cv::mean(GreenChannel)[0];
    redAvg = cv::mean(RedChannel)[0];

    double K_sum = (blueAvg + greenAvg + redAvg)/3.0;
    double Kb = K_sum/blueAvg;
    double Kg = K_sum/greenAvg;
    double Kr = K_sum/redAvg;

    cv::addWeighted(BlueChannel,Kb,0,0,0,BlueChannel);
    cv::addWeighted(GreenChannel,Kg,0,0,0,GreenChannel);
    cv::addWeighted(RedChannel,Kr,0,0,0,RedChannel);

    cv::merge(vchannels,dest);

}


cv::Rect drawText(cv::Mat &img, string text, cv::Point coord, cv::Scalar color, float fontScale, int thicness, int fontFace )
{
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, fontFace,fontScale,thicness,&baseline);

    // adjust coordinates
    if(coord.y >= 0)
        coord.y += textSize.height;
    else
        coord.y += img.rows - baseline +1;

    if(coord.x < 0)
        coord.x += img.cols - textSize.width +1;
    cv::Rect boundRect = cv::Rect(coord.x, coord.y - textSize.height, textSize.width, baseline +textSize.height);
    cv::putText(img,text,coord,fontFace,fontScale,color,thicness);

    return boundRect;
}

cv::Rect drawButton(cv::Mat &img, string text, cv::Point coord, int minWidth)
{
    int B = cv::BORDER_DEFAULT;
    //get box Rect
    cv::Point textCoord = cv::Point(coord.x+B,coord.y +B);
    //get Text Rect
    cv::Rect rcText = drawText(img,text,textCoord,cv::Scalar(0,0,0));

    cv::Rect rcButton = cv::Rect(rcText.x-B,rcText.y-B,rcText.width+2*B,rcText.height+2*B);
    if(rcButton.width < minWidth)
        rcButton.width = minWidth;
    cv::Mat matButton = img(rcButton);
    matButton += cv::Scalar(90,90,90);

    cv::rectangle(img,rcButton,cv::Scalar(200,200,200),1,cv::LINE_AA);
    drawText(img,text,textCoord,cv::Scalar(10,55,20));

    return rcButton;

}

bool isButtonRect(const cv::Point pt, const cv::Rect rect)
{
    if( pt.x >= rect.x && pt.x <= (rect.x+rect.width-1))
        if(pt.y >= rect.y && pt.y <= (rect.y + rect.height -1)) return true;
    return false;
}
