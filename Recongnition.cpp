#include"Recognition.hpp"

const int N_COMPONTS = 10;

const double THRESHOLD = 1230.0;



cv::Ptr<cv::face::FaceRecognizer>    recognitionFace::learnCollectedFacesAndTrain(INPUT const std::vector<cv::Mat> &preprocFaces,
                                                      OUTPUT const std::vector<int> &labels,
                                                      OUTPUT const FACE_REC_METHOD facerecAlgorithm)
{
    if(facerecAlgorithm == FISHERFACE)
    {
        //set componts and a thresthod for PCA
        model = cv::face::createFisherFaceRecognizer(N_COMPONTS,THRESHOLD);
        if(model.empty())
        {
            std::cerr<<"ERROR: The FaceRecognizer algorithm ("<<facerecAlgorithm<<")is not available!"<<std::endl;
            exit(1);
        }
        //model->train(preprocFaces,labels);

    }else if(facerecAlgorithm == EIGENFACES)
    {
        model == cv::face::createEigenFaceRecognizer(0,THRESHOLD);
        if(model.empty())
        {
            std::cerr<<"ERROR: The FaceRecognizer algorithm ("<<facerecAlgorithm<<")is not available!"<<std::endl;
            exit(1);
        }
    }else {
        std::cerr<<"ERROR: not algorithm ("<<facerecAlgorithm<<") is available"<<std::endl;
        exit(1);
    }
    model->train(preprocFaces,labels);
    this->haveModel = true;
    return model;
}

bool recognitionFace::predict(INPUT Mat *face, OUTPUT int &predictLabel, OUTPUT double &confidence)
{
    if(true == haveModel)
    {
        model->predict(*face, predictLabel , confidence);
        return true;
    }else {
       return false;
    }
}

recognitionFace::recognitionFace()
{
    this->haveModel = false;
}






 cv::Ptr<cv::face::FaceRecognizer>  recognitionFace::getModel(){
     return model;
 }

cv::Mat getImageFrom1DMat(const cv::Mat matrixRow, int height)
{
    cv::Mat rectangularMat = matrixRow.reshape(1,height);
    Mat dst;
    cv::normalize(rectangularMat,dst,0,255,CV_8UC1);
    return dst;
}


double getSimilaryity(const Mat A, const Mat B)
{
    if(A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols)
    {
        double errorL2 =cv::norm(A,B,cv::NORM_L2);
        double similarity = errorL2/(A.rows*A.cols);
        return similarity;
    }else
        return 10000000.0;

}

cv::Mat reconstructFace(const cv::Ptr<cv::face::FaceRecognizer> model, const cv::Mat perprocessedFace)

{
    try{
        cv::Mat eigenvectors = model->get<cv::Mat>("eigenvectors");
        cv::Mat averageFaceRow = model->get<cv::Mat>("mean");
        int faceHeight = perprocessedFace.rows;

        cv::Mat projection = cv::LDA::subspaceProject(eigenvectors,averageFaceRow,perprocessedFace.reshape(1,1));
        Mat reconstructionRow = cv::LDA::subspaceReconstruct(eigenvectors, averageFaceRow, projection);

        cv::Mat reconstructionMat = reconstructionRow.reshape(1, faceHeight);
                // Convert the floating-point pixels to regular 8-bit uchar pixels.
        cv::Mat reconstructedFace = cv::Mat(reconstructionMat.size(), CV_8U);
        reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);
                //printMatInfo(reconstructedFace, "reconstructedFace");

        return reconstructedFace;
    }catch(cv::Exception e){
        return cv::Mat();
    }

}


