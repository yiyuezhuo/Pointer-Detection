# Ref to https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/

import cv2 as cv
import os
import numpy as np
import glob
import os

confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image


import matplotlib.patches as patches
import matplotlib.pyplot as plt

def rectangle(xy, width, height, edgecolor='r', linewidth=1, facecolor='none'):
    rect = patches.Rectangle(xy, width, height, linewidth=linewidth, 
                             edgecolor=edgecolor, facecolor=facecolor)
    plt.gca().add_patch(rect)

def show_detection(img, boxes, show=True, text_list = None,
                   fontsize = 12):
    '''
    img: RGB image
    boxes: List of (xmin,ymin,xmax,ymax). For shorthand for (left,top,width,height), use show_detection2
    '''
    plt.imshow(img)
    for i,box in enumerate(boxes):
        xmin,ymin,xmax,ymax = box
        #print((xmin,ymin), xmax-xmin, ymax-ymin)
        rectangle((xmin,ymin), xmax-xmin, ymax-ymin)
        if text_list is not None:
            plt.text(xmin, ymin-30, text_list[i], fontsize=fontsize, bbox=dict(facecolor='purple', alpha=0.1))
    if show:
        plt.show()
#%matplotlib inline

def show_detection2(img, boxes_ltwh, **kwargs):
    boxes_ltrb = []
    for left,top,width,height in boxes_ltwh:
        boxes_ltrb.append([ left, top, left + width, top + height])
    show_detection(img, boxes_ltrb, **kwargs)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load names of classes
classesFile = "voc.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
#modelConfiguration = os.path.abspath("yolov3-tiny-tank.cfg")
#modelWeights = os.path.abspath("backup_YOLO/yolov3-tiny-tank_19000.weights")
modelConfiguration = "yolov3-tiny-pointer.cfg"
modelWeights = "backup_YOLO/yolov3-tiny-pointer_19000.weights"
#modelWeights = "backup_YOLO/yolov3-tiny-tank_21000.weights"
#modelWeights = "backup_YOLO/yolov3-tiny-tank_30000.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)



def detect(frame):
    
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1.0/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
    
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                
    return classIds,confidences,boxes

def preprocess(name=None, img_path=None, transform=None):
    '''
    preprocess wrap the loading and detecting operation done by network.
    '''
    #name = '20180421153245.jpg' # multiple box
    if not img_path:
        img_path = os.path.join('data', name)
    frame_bgr = cv.imread(img_path)
    if frame_bgr is None:
        raise Exception("wtf OpenCV can't get image")
    frame = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)

    if transform:
        frame = transform(frame)

    classIds,confidences,boxes = detect(frame)
    return frame,classIds,confidences,boxes

def save_box_fig(frame, boxes, target_path, figsize = None):
    boxes_ltwh = boxes
    boxes_ltrb = []
    for left,top,width,height in boxes_ltwh:
        boxes_ltrb.append([ left, top, left + width, top + height])

    if figsize:
        fig = plt.figure(figsize=figsize)

    show_detection(frame, boxes_ltrb, show=False)
    
    plt.savefig(target_path)
    plt.clf()
    
    if figsize:
        plt.close(fig)

    

def batch_detect(pattern, target_dir, figsize=None, savefig=True, callback=None):
    for img_path in glob.glob(pattern):
        frame_bgr = cv.imread(img_path)
        frame = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)

        target_path = os.path.join(target_dir, os.path.split(img_path)[1])

        classIds,confidences,boxes = detect(frame)

        if callback:
            callback(frame=frame, 
                        classIds=classIds,
                        confidences=confidences, 
                        boxes=boxes,
                        img_path=img_path,
                        target_path=target_path)

        if savefig:
            save_box_fig(frame, boxes, target_path, figsize=figsize)

def extract_max(classIds, confidences):
    max_dial_idx = None # 还是学C++的写法吧。。反正等下还要翻译成C++，这个写法本身的numpy写法也不算直观
    max_pointer_idx = None
    max_dial_conf = 0
    max_pointer_conf = 0
    for i in range(len(classIds)):
        if classIds[i] == 0:
            if confidences[i] > max_dial_conf:
                max_dial_conf = confidences[i]
                max_dial_idx = i
        elif classIds[i] == 1:
            if confidences[i] > max_pointer_conf:
                max_pointer_conf = confidences[i]
                max_pointer_idx = i
    return max_dial_idx, max_pointer_idx

def parse_pointer_all(frame, dial_box, pointer_box):
    pt = pointer_box
    
    base_angle = np.arctan(pt[3]/pt[2])
    
    # adjust cuted region
    
    print(dial_box)
    print(dial_box[1] > frame.shape[0] * 0.9)
    print(dial_box[3] < dial_box[2])
    
    if dial_box[0] < frame.shape[1] * 0.1 and dial_box[2] < dial_box[3]:
        dial_box = (dial_box[0] - (dial_box[3] - dial_box[2]), dial_box[1], dial_box[3], dial_box[3])
    elif (dial_box[0]+dial_box[2]) > frame.shape[1] * 0.9 and dial_box[2] < dial_box[3]:
        dial_box = (dial_box[0] , dial_box[1], dial_box[3], dial_box[3])
    elif dial_box[1] < frame.shape[0] * 0.1 and dial_box[3] < dial_box[2]:
        dial_box = (dial_box[0], dial_box[1] - (dial_box[2] - dial_box[3]), dial_box[2], dial_box[2])
    elif (dial_box[1]+dial_box[3]) > frame.shape[0] * 0.9 and dial_box[3] < dial_box[2]:
        dial_box = (dial_box[0], dial_box[1], dial_box[2], dial_box[2])
        
    print(dial_box)
    
    if pt[0]+pt[2]//2 < dial_box[0]+dial_box[2]//2: # point to left
        if pt[1]+pt[3]//2 < dial_box[1]+dial_box[3]//2: # point to top
            dir_idx = 0
        else:
            dir_idx = 1
    else:
        if pt[1]+pt[3]//2 < dial_box[1]+dial_box[3]//2:
            dir_idx = 2
        else:
            dir_idx = 3
    
    if dir_idx ==0:
        tail = (pt[0]+pt[2],pt[1]+pt[3])
        head = (pt[0],pt[1])
        angle = -np.pi/2 - (np.pi/2-base_angle)
    elif dir_idx == 1:
        tail = (pt[0]+pt[2],pt[1])
        head = (pt[0],pt[1]+pt[3])
        angle = -np.pi-base_angle
    elif dir_idx == 2:
        tail = (pt[0],pt[1]+pt[3])
        head = (pt[0]+pt[2],pt[1])
        angle = -base_angle
    elif dir_idx == 3:
        tail = (pt[0],pt[1])
        head = (pt[0]+pt[2],pt[1]+pt[3])
        angle = base_angle
    
    print(['lefttop', 'leftbottom', 'righttop', 'rightbottom'][dir_idx])
                    
    return angle, tail, head


def process_all(frame, classIds, confidences, boxes, thickness=3):
    '''
    frame_line = process_all(frame, classIds, confidences, boxes)
    plt.figure(figsize=(16,9))
    plt.imshow(frame_line)
    '''

    max_dial_idx, max_pointer_idx = extract_max(classIds, confidences)
    
    if max_dial_idx is None or max_pointer_idx is None:
        return frame.copy()
    
    dial_box = boxes[max_dial_idx]
    pointer_box = boxes[max_pointer_idx]
    
    angle, tail, head = parse_pointer_all(frame, dial_box, pointer_box)
    print(angle,tail,head)
    
    frame_line = frame.copy()
    cv.line(frame_line, tail, head, (255, 0, 255), thickness, 8);
    #cv.line(frame_line, tail_new, head_new, (0, 255, 0), thickness, 8);
    
    point = (angle/(np.pi/2))*(0.74-0.4) + 0.74
    
    cv.putText(frame_line, str(point), (head[0],head[1]-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 23, 0), 4, 8)
    
    return frame_line

def show_YOLO(img_path, label_path):
    frame = cv.imread(img_path)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_width = frame.shape[1]
    img_height = frame.shape[0]
    with open(label_path) as f:
        sl = f.read()
    for line in sl.split('\n'):
        if len(line) == 0:
            continue
        class_id, centerx, centery, width, height = line.strip().split(' ')
        class_id = int(class_id)
        centerx, centery, width, height = [float(n) for n in [centerx,centery,width,height]]
        left = centerx - width/2
        top = centery - height/2
        left = int(left * img_width)
        top = int(top * img_height)
        width = int(width * img_width)
        height = int(height * img_height)
        
        cv.rectangle(frame, (left,top),(left+width,top+height), (255,0,0), 2)
    
    plt.figure(figsize=(16,9))
    plt.imshow(frame)

def YOLO_label_decode(doc, img_width, img_height):
    rect_list = []
    for line in doc.split('\n'):
        line = line.strip()
        if len(line) == 0:
            continue
        class_id, centerx, centery, width, height = line.split(' ')
        class_id = int(class_id)
        centerx, centery, width, height = [float(n) for n in [centerx,centery,width,height]]
        left = centerx - width/2
        top = centery - height/2
        left = int(left * img_width)
        top = int(top * img_height)
        width = int(width * img_width)
        height = int(height * img_height)
        rect_list.append([class_id, left, top, width, height])
    return rect_list

def YOLO_label_encode(rect_list, img_width, img_height):
    lines = []
    for class_id, left, top, width, height in rect_list:
        centerx = (left + left + width)/2.0
        centery = (top  + top + height)/2.0
        
        centerx /= img_width
        centery /= img_height
        width /= img_width
        height /= img_height
        
        # https://github.com/ssaru/convert2Yolo/blob/master/Format.py#L592
        line = '{} {} {} {} {}'.format(class_id, round(centerx,3), round(centery,3), round(width,3), round(height,3))
        lines.append(line)
    doc = '\n'.join(lines) + '\n'
    return doc


def rotate_img_label(frame_source, label_source, frame_target, label_target, angle):
    #这个函数假设了只从只指向右上角的原始图像中读取，不过已经投入的加强图像.angle是角度制
    
    # decode
    frame = cv.imread(frame_source)
    with open(label_source) as f:
        doc = f.read()
    rects = YOLO_label_decode(doc, frame.shape[1], frame.shape[0])
    
    dial = None
    pt = None
    
    for class_id, left, top, width, height in rects:
        if class_id == 0:
            dial = (left, top, width, height)
        elif class_id == 1:
            pt = (left, top, width, height)
        else:
            raise Exception("Unknown object")
            
    if dial is None or pt is None:
        print('skip {}'.format(frame_source)) # 那种很模糊的，只标了dial的图像
        return None
        
    r = max(dial[2], dial[3])
    tail = (pt[0],pt[1]+pt[3])
    head = (pt[0]+pt[2],pt[1])
        
    
    # transform
    
    rotate_center = tail #tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(rotate_center, angle, 1.0)
    result = cv.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv.INTER_LINEAR)
    
    tail_new = tuple((rot_mat @ np.concatenate([tail,[1.0]])).astype(int).tolist())
    head_new = tuple((rot_mat @ np.concatenate([head,[1.0]])).astype(int).tolist())
    
    pt_new_lefttop = (min(tail_new[0], head_new[0]), min(tail_new[1], head_new[1]))
    pt_new_rightbottom = (max(tail_new[0], head_new[0]), max(tail_new[1], head_new[1]))

    # encode
    
    dial_new = (tail_new[0]-r//2, tail_new[1]-r//2, r, r)
    pt_new = (pt_new_lefttop[0], pt_new_lefttop[1], 
              pt_new_rightbottom[0] - pt_new_lefttop[0], pt_new_rightbottom[1] - pt_new_lefttop[1])
    rect_list = [(0,)+dial_new, (1,)+pt_new]
    doc_new = YOLO_label_encode(rect_list, result.shape[1], result.shape[0])
    
    cv.imwrite(frame_target, result)
    with open(label_target, 'w') as f:
        f.write(doc_new)





print('OpenCV version: {}'.format(cv.__version__))
print("load {} {}".format(modelConfiguration, modelWeights))
print(classes)
