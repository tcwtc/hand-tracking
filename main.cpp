
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video/tracking.hpp>

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <deque>

using namespace std;
using namespace cv;

#define PATH_BUFFER 30



class CascadeDetectorAdapter: public DetectionBasedTracker::IDetector
{
public:
    CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector):
            IDetector(),
            Detector(detector)
    {
        CV_Assert(detector);
    }

    void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects)
    {
        Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
    }

    virtual ~CascadeDetectorAdapter()
    {}

private:
    CascadeDetectorAdapter();
    cv::Ptr<cv::CascadeClassifier> Detector;
};


bool sort_Rect_withArea(const Rect & r1, const Rect & r2)
{
    return r1.width*r1.height < r2.width*r2.height;
}

void trackPath(deque<Point> pts, unsigned long int frameID, Point & Mv_Dire, Mat & frame){
    for(size_t i = 1; i < PATH_BUFFER; ++i){
        if(pts.at(i-1)==Point(0,0) or pts.at(i)==Point(0,0))
            continue;
        if( frameID>=10 and i==1 and pts.at(PATH_BUFFER-10)!=Point(0,0) ){
            Point dPoint;
            dPoint.x = pts[PATH_BUFFER-10].x - pts[i].x;
            dPoint.y = pts[PATH_BUFFER-10].y - pts[i].y;

            Mv_Dire.x = (abs(dPoint.x)>20)?1:-1;
            Mv_Dire.y = (abs(dPoint.y)>20)?1:-1;
        }

//        int thickness = sqrt(PATH_BUFFER / float(i + 1) * 2.5;
        int thickness = 5;
        line(frame, pts[i - 1], pts[i], CV_RGB(0, 0, 255), thickness);
    }
}


int main(int , char** )
{
    const string WindowName = "Palm Detection example";

    namedWindow(WindowName);

    VideoCapture VideoStream(0);
    unsigned long int frameID = 0;

    if (!VideoStream.isOpened())
    {
        printf("Error: Cannot open video stream from camera\n");
        return 1;
    }

    string cascadeFrontalfilename = "../xml/palm.xml";
    Ptr<CascadeClassifier> cascade = makePtr<CascadeClassifier>(cascadeFrontalfilename);
    Ptr<DetectionBasedTracker::IDetector> MainDetector = makePtr<CascadeDetectorAdapter>(cascade);
    if ( cascade->empty() )
    {
        printf("Error: Cannot load %s\n", cascadeFrontalfilename.c_str());
        return 2;
    }

    cascade = makePtr<	CascadeClassifier>(cascadeFrontalfilename);
    Ptr<DetectionBasedTracker::IDetector> TrackingDetector = makePtr<CascadeDetectorAdapter>(cascade);
    if ( cascade->empty() )
    {
        printf("Error: Cannot load %s\n", cascadeFrontalfilename.c_str());
        return 2;
    }

    DetectionBasedTracker::Parameters params;
    DetectionBasedTracker Detector(MainDetector, TrackingDetector, params);

    if (!Detector.run())
    {
        printf("Error: Detector initialization failed\n");
        return 2;
    }

    Mat ReferenceFrame;
    Mat GrayFrame;
    vector<Rect> Palms;
    Rect selectedPalm;
    Point selectedCenter;

    deque<Point> pts(PATH_BUFFER);
    deque<Point> predict_pts(PATH_BUFFER);
    Point Mv_Dire(0,0);
    // Move direction, x = 1 for east, -1 for west
    // y = -1 for south, +1 for north
    string direction;

    VideoStream >> ReferenceFrame;
    Size s = ReferenceFrame.size();

// Init KF
    KalmanFilter KF(4, 2, 0);
    KF.transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    Mat_<float> measurementPT(2,1); measurementPT.setTo(Scalar(0));
    KF.statePre.at<float>(0) = s.width*0.5;
    KF.statePre.at<float>(1) = s.height*0.5;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(10));
    setIdentity(KF.errorCovPost, Scalar::all(.1));
    Mat predictionTrack, estimatedTrack;
// End of Init KF

    do
    {
        VideoStream >> ReferenceFrame;
        cvtColor(ReferenceFrame, GrayFrame, COLOR_RGB2GRAY);
        Detector.process(GrayFrame);
        Detector.getObjects(Palms);

        predictionTrack = KF.predict();
//        Point predictPT(predictionTrack.at(0),predictionTrack.at(1));

        if(Palms.size()>0){
            sort(Palms.begin(), Palms.end(), sort_Rect_withArea);
            int radius = (Palms[0].width+Palms[0].height)*0.5;

            if(radius>20){
                rectangle(ReferenceFrame, Palms[0], Scalar(0,255,0));
                selectedPalm = Palms[0];
                selectedCenter.x = selectedPalm.x + 0.5*selectedPalm.width;
                selectedCenter.y = selectedPalm.y + 0.5*selectedPalm.height;
                // add to queue
                pts.push_front(selectedCenter);
                pts.pop_back();
            }
            trackPath(pts, frameID, Mv_Dire, ReferenceFrame);
            //handle when both directions are non-empty
            if(Mv_Dire.x!=0 and Mv_Dire.y!=0){
                direction = to_string(Mv_Dire.x)+", "+to_string(Mv_Dire.y);
            }
            else{
                direction = (Mv_Dire.x!=0)?to_string(Mv_Dire.x):to_string(Mv_Dire.y);
            }
            putText(ReferenceFrame, direction, Point(30,30), FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 5);
//            putText(ReferenceFrame, "dx: {}, dy: {}".format(dX, dY),\
//             (10, ReferenceFrame.shape[0] - 10), FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1);

            measurementPT(0) = selectedCenter.x;
            measurementPT(1) = selectedCenter.y;

            estimatedTrack = KF.correct(measurementPT);
            Point estimatedPoint(estimatedTrack.at<float>(0),estimatedTrack.at<float>(1));
            predict_pts.push_front(estimatedPoint); predict_pts.pop_back();
            for(size_t i = 1; i < PATH_BUFFER; ++i){
                if(predict_pts.at(i-1)==Point(0,0) or predict_pts.at(i)==Point(0,0))
                    continue;
                int thickness = 5;
                line(ReferenceFrame, predict_pts[i - 1], predict_pts[i], CV_RGB(255, 0, 0), thickness);
            }

        }

        resize(ReferenceFrame, ReferenceFrame, Size(), 0.5, 0.5, INTER_LINEAR);
        imshow(WindowName, ReferenceFrame);
        frameID += 1;

    } while (waitKey(30) > 0);

    Detector.stop();

    return 0;
}
