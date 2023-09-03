import numpy as np
import cvzone
import math
from ultralytics import YOLO
import cv2;
from sort.sort import *
from util import get_car,read_license_plate,write_csv
import util
#test vedio soruce
#https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/


#resutls
results={};
#Tracking object.
mot_tracker=Sort();
#load model
coco_model=YOLO('yolov8n.pt');
license_plate_detector=YOLO("license_plate_detector.pt");

#load vedio
cap=cv2.VideoCapture('Untitled video - Made with Clipchamp.mp4');
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
if not cap.isOpened():
  print('Could not open video file')
  exit()


# Get the video frame size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize the window to the size of the video frame
cv2.resizeWindow('Video', width, height)


#reading classes:
classfile='coco.names';
classNames=[];
with open(classfile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n');
print(classNames);



#object classes
vehicles=[2,3,5,7];
#read frames
frame_nmr=-1;
ret=True;
while ret:
    frame_nmr+=1;
    ret,frame=cap.read();
    if not ret:
        print('Could not read frame')
        write_csv(results, './test.csv');

    #detect vehicles
    if ret :
        results[frame_nmr]={};
        detections = coco_model(frame)[0]
        detections_=[];
        for detection in detections.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id=detection;
            conf = math.ceil((detection[4] * 100)) / 100;
            cls = int(detection[5]);
            currentClass = classNames[cls];
            if currentClass == "car" or currentClass == "bus" or currentClass == "truck" or currentClass == "motorbike" and conf > 0.5:
                detections_.append([x1,y1,x2,y2,score]);

        #tracking objects
        track_ids=mot_tracker.update(np.asarray(detections_));
        for result in track_ids:
            x1, y1, x2, y2, Id = result;
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2);
            w, h = x2 - x1, y2 - y1;
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3);
            cvzone.putTextRect(frame, f'{int(Id)   }', (max(0, x1), max(35, y1)), offset=5, thickness=2,
                               scale=1);


        #license plate detection
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id=license_plate;
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2);

            #Assign detected license plates to car
            if score>0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3);
                cvzone.putTextRect(frame, 'Nplate', (max(0, x1), max(35, y1)), offset=5, thickness=2,
                                   scale=1);
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids);

            if car_id!=-1:
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV);

                # Read license plate numbers
                license_plate_text, license_plat_text_score = read_license_plate(license_plate_crop_thresh);
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score ,
                                                                    'text_score': license_plat_text_score}}
        #Display the frame
    cv2.imshow('Video', frame)

    # Wait for the user to press a key
    cv2.waitKey(1);























