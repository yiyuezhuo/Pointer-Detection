/*
Require:
    OpenCV 3.4.2 for support to darknet(YOLOv3)
Author:
    yiyuezhuo
Reference:
    https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
    https://github.com/spmallick/learnopencv/blob/master/ObjectDetection-YOLO/object_detection_yolo.cpp
    https://github.com/AlexeyAB/darknet
    https://github.com/pjreddie/darknet
Compile command example:
    1. Open your VC developing tool command line for x64 to config enviroment varible.
        For example: 
        D:\VisualStudio\VC\Auxiliary\Build\vcvars64.bat
        Or in VC menu you can see it.
    2. Now cl command is available in your cmd, input command such as:
        cl pointer_detection.cpp opencv_world342.lib -IE:\agent2\opencv-release-342\opencv\build\include /link /OUT:"test.exe" /SUBSYSTEM:CONSOLE /MACHINE:X64 /LIBPATH:E:\agent2\opencv-release-342\opencv\build\x64\vc14\lib
        Modify include path(-I) for compiler cl and /LIBPATH for linker to search OpenCV include and lib directory.
*/

#define DETECT_DEBUG
#define USE_MINI_DATASET


#include<vector>
#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
         
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
         
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}


// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs, 
    vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes)
{
    /*
    // Export them
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    */
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
}

int detect_dial_pointer(Net net, Mat frame, Rect& rect_dial, Rect& rect_pointer){ // frame is just a image mat, I am tired to rename them.
    Mat blob;
    vector<int> classIds; 
    vector<float> confidences;
    vector<Rect> boxes;


    // Create a 4D blob from a frame.
    blobFromImage(frame, blob, 1/255.0, cvSize(inpWidth, inpHeight), Scalar(0,0,0), true, false);
     
    //Sets the input to the network
    net.setInput(blob);
     
    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Remove the bounding boxes with low confidence
    postprocess(frame, outs, classIds, confidences, boxes);

    // extract max
    int max_dial_idx = -1;
    int max_pointer_idx = -1;
    float max_dial_conf = 0.0;
    float max_pointer_conf = 0.0;

    for(size_t i=0; i<classIds.size(); i++){
        if(classIds[i] == 0){
            if(confidences[i] > max_dial_conf){
                max_dial_conf = confidences[i];
                max_dial_idx = i;
            }
        }
        else if (classIds[i] == 1){
            if(confidences[i] > max_pointer_conf){
                max_pointer_conf = confidences[i];
                max_pointer_idx = i;
            }
        }
    }

    #ifdef DETECT_DEBUG
    
    if(max_dial_idx != -1)
        rectangle(frame, boxes[max_dial_idx], Scalar(255, 0, 0), 2);
    if(max_pointer_idx != -1)
        rectangle(frame, boxes[max_pointer_idx], Scalar(128, 0, 0), 2);

    #endif

    if((max_dial_idx == -1) || (max_pointer_idx == -1)){
        cout << "fail to detect dial or pointer" << endl;
        return 1;
    }

    rect_dial = boxes[max_dial_idx];
    rect_pointer = boxes[max_pointer_idx];
    
    return 0;
}

float parse_pointer_all(Mat frame, Rect dial_box, Rect pointer_box, Point& tail, Point& head){
    Rect pt = pointer_box;

    float angle;
    float base_angle = atan((float)(pt.height)/(float)(pt.width));

    // adjust cuted edge
    if ((dial_box.x < frame.cols * 0.1) && (dial_box.width < dial_box.height)){
        dial_box = Rect(dial_box.x - (dial_box.height - dial_box.width), dial_box.y, dial_box.height, dial_box.height);
    }
    else if (((dial_box.x+dial_box.width) > frame.cols * 0.9) && (dial_box.width < dial_box.height)){
        dial_box = Rect(dial_box.x , dial_box.y, dial_box.height, dial_box.height);
    }
    else if ((dial_box.y < frame.rows * 0.1) && (dial_box.height < dial_box.width)){
        dial_box = Rect(dial_box.x, dial_box.y - (dial_box.width - dial_box.height), dial_box.width, dial_box.width);
    }
    else if (((dial_box.y+dial_box.height) > frame.rows * 0.9) && (dial_box.height < dial_box.width)){
        dial_box = Rect(dial_box.x, dial_box.y, dial_box.width, dial_box.width);
    }

    
    if(pt.x+pt.width/2 < dial_box.x+dial_box.width/2){
        cout << "left" << endl;
        if(pt.y+pt.height/2 < dial_box.y+dial_box.height/2){
            cout << "top" << endl;
            tail.x = pt.x+pt.width;
            tail.y = pt.y+pt.height;
            head.x = pt.x;
            head.y = pt.y;
            angle = -CV_PI/2 - (CV_PI/2 - base_angle);
        }
        else{
            cout << "bottom" << endl;
            tail.x = pt.x+pt.width;
            tail.y = pt.y;
            head.x = pt.x;
            head.y = pt.y+pt.height;
            angle = -CV_PI - base_angle;
        }
    }
    else{
        cout << "right" << endl;
        if(pt.y+pt.height/2 < dial_box.y+dial_box.height/2){
            cout << "top" << endl;
            tail.x = pt.x;
            tail.y = pt.y+pt.height;
            head.x = pt.x+pt.width;
            head.y = pt.y;
            angle = -base_angle;
        }
        else{
            cout << "bottom" << endl;
            tail.x = pt.x;
            tail.y = pt.y;
            head.x = pt.x+pt.width;
            head.y = pt.y+pt.height;
            angle = base_angle;
        }
    }

    cout << "base_angle:" << base_angle << " angle:" << angle << endl;
    return angle;
}

float process_all(Mat frame, Rect dial_box, Rect pointer_box){
    Point tail;
    Point head;
    float angle;
    float point;
    
    angle = parse_pointer_all(frame, dial_box, pointer_box, tail, head);
    point = (angle/(CV_PI/2))*(0.74-0.4) + 0.74;

    cout << "angle:" << angle << " tail:" << tail << " head:" << head << endl;

    //cout << "thickness:" << thickness << endl;
    line(frame, tail, head, Scalar(255, 0, 255), 3, 8);
    //cout << "draw line end";
    putText(frame, to_string(point), Point(head.x,head.y-12), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 4, 8);

    //cout << "draw end" << endl;
    return point;
}

int detect(Net net, Mat frame, Rect& rect_dial, Rect& rect_pointer, float& point){

    int is_fail = detect_dial_pointer(net, frame, rect_dial, rect_pointer);
    if(is_fail)
        return 1;
    
    point = process_all(frame, rect_dial, rect_pointer);
    // There may exist some image processing methods, but they're not robust enough for this task.

    return 0;
}

int single_test(Net net, Mat frame, float& point){
    Rect rect_dial;
    Rect rect_pointer;

    int detect_fail = detect(net, frame, rect_dial, rect_pointer, point);
    if(detect_fail){
        //cout << "fail: " << path << endl;
        return 1;
    }

    return 0;
}

void batch_test(Net net, String source_dir, String target_dir){
    String pattern = source_dir + "/*.jpg";
    String source_path,target_path;
    vector<String> file_vector;
    int num_succ = 0;
    int num_fail = 0;
    float point;
    
    glob(pattern, file_vector, false);
    for(size_t i=0; i<file_vector.size(); i++){
        source_path = file_vector[i];
        // Too bad `os.listdir` and `os.path.split`, two naive functions can not be easily mirrored in C++.
        // https://stackoverflow.com/questions/35530092/c-splitting-an-absolute-file-path
        // TODO: use macro to provide cross-plateform ability
        // For linux  
        //std::size_t botDirPos = source_path.find_last_of("/");
        //target_path = target_dir + '/' + source_path.substr(botDirPos, source_path.length());
        // Fow windows
        std::size_t botDirPos = source_path.find_last_of("\\");
        target_path = target_dir + "\\" + source_path.substr(botDirPos, source_path.length());


        Mat frame = imread(source_path);
        
        cout << "processing" << source_path << endl;
        
        int detect_fail = single_test(net, frame, point); // inplace modify frame
        if(detect_fail){
            cout << "fail: " << source_path << endl;
            num_fail++;
        }
        else{
            cout << "succ: " << source_path << " " << point << endl;
            num_succ++;
        }
        
        cout << "output:" << target_path << endl;
        imwrite(target_path, frame);

        #ifdef DETECT_DEBUG
        
        if(detect_fail){
            String debug_path = "debug_cpp\\" + source_path.substr(botDirPos, source_path.length());
            cout << "debug_path=" << debug_path << endl;
            imwrite(debug_path, frame);

            Mat frame_origin = imread(source_path);
            String debug_raw_path = "debug_cpp_raw\\" + source_path.substr(botDirPos, source_path.length());
            imwrite(debug_raw_path, frame_origin);
        }

        #endif
    }
    cout << "fail: " << num_fail << " succ: " << num_succ << "succ ratio" << (float)(num_succ)/((float)(num_succ)+(float)(num_fail));
}

int main(){
    // Load names of classes
    string classesFile = "voc.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);
    
    // Give the configuration and weight files for the model
    String modelConfiguration = "yolov3-tiny-pointer.cfg";
    //String modelWeights = "yolov3-tiny-tank_21000.weights";
    String modelWeights = "yolov3-tiny-pointer_24000.weights";
    
    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    cout << "model load load succ from: " << modelWeights << endl;

    /*
    String path = "E:\\agent4\\lab3\\tank\\data\\20180413152222.jpg";
    Mat frame = imread(path);
    Rect refined_piece;
    float score;
    int detect_fail = detect(net, frame, refined_piece, score);
    if(detect_fail){
        cout << "fail: " << path << endl;
        return 1;
    }

    rectangle(frame, refined_piece, Scalar(0, 255, 0), 2);
    putText(frame, to_string(score), Point(refined_piece.x-10, refined_piece.y-10), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 4, 8);
    imwrite("predicted.png", frame);
    cout << "succ: " << path << endl;
    */
    #ifdef USE_MINI_DATASET
    batch_test(net, "mini_data", "mini_results");
    #endif

    #ifndef USE_MINI_DATASET
    batch_test(net, "data", "results");
    #endif

    return 0;
}