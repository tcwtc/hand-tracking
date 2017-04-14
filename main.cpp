
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

const string WindowName = "Palm Detection example";

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

int main(int , char** )
{
    namedWindow(WindowName);

    VideoCapture VideoStream(0);

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

    do
    {
        VideoStream >> ReferenceFrame;
        cvtColor(ReferenceFrame, GrayFrame, COLOR_RGB2GRAY);
        Detector.process(GrayFrame);
        Detector.getObjects(Palms);

        // for (size_t i = 0; i < Palms.size(); i++)
        // {
        //     rectangle(ReferenceFrame, Palms[i], Scalar(0,255,0));
        // }
        // if(Palms.size()>0){
        //     std::cout<<Palms[0].width<<std::endl;
        // }
        if(Palms.size()>0){
            sort(Palms.begin(), Palms.end(), sort_Rect_withArea);
            int radius = (Palms[0].width+Palms[0].height)*0.5;

            if(radius>20){
                rectangle(ReferenceFrame, Palms[0], Scalar(0,255,0));
                // add to queue
            }
        }

        imshow(WindowName, ReferenceFrame);

    } while (waitKey(30) > 0);

    Detector.stop();

    return 0;
}
