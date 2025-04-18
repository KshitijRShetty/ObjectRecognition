#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace dnn;

vector<string> loadLabels(const string& filename) {
    vector<string> labels;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
}

void main() {
    VideoCapture video(0);
    CascadeClassifier faceCascade, eyeCascade;
    Mat img;

    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        cout << "Error loading face cascade file." << endl;
        return;
    }
    if (!eyeCascade.load("haarcascade_eye.xml")) {
        cout << "Error loading eye cascade file." << endl;
        return;
    }

    Net net = readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel");
    vector<string> labels = loadLabels("object_labels.txt");

    while (true) {
        video.read(img);
        if (img.empty()) break;

        Mat blob = blobFromImage(img, 0.007843, Size(300, 300), 127.5);
        net.setInput(blob);
        Mat detections = net.forward();

        Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

        vector<string> detectedObjectNames;

        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);

            if (confidence > 0.5) {
                int classId = static_cast<int>(detectionMat.at<float>(i, 1));
                int left = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
                int top = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
                int right = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
                int bottom = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);

                rectangle(img, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);

                string label = labels[classId] + ": " + to_string(static_cast<int>(confidence * 100)) + "%";
                detectedObjectNames.push_back(label);
                putText(img, label, Point(left, top - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
            }
        }

     
        int sidePanelWidth = 250;
        Mat sidePanel(img.rows, sidePanelWidth, img.type(), Scalar(30, 30, 30));
        int y = 40;  

      
        putText(sidePanel, "Detected Objects", Point(10, y), FONT_HERSHEY_DUPLEX, 0.8, Scalar(255, 255, 255), 1);
        y += 40;  

        for (const auto& name : detectedObjectNames) {
            
            rectangle(sidePanel, Point(5, y - 25), Point(sidePanelWidth - 5, y + 5), Scalar(70, 70, 70), FILLED);
            
            putText(sidePanel, name, Point(15, y), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
            y += 35;  
        }

        
        Mat displayFrame;
        hconcat(img, sidePanel, displayFrame);

        imshow("Frame with Object Names", displayFrame);
        if (waitKey(1) == 27) break; 
    }
}
