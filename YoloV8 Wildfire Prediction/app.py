from ultralytics import YOLO
import cv2
import cvzone 
import math

model = YOLO('best.pt')
#If you want to use webcam
# model.predict(source=0, imgsz= 800, conf = 0.6, show = True)


cap = cv2.VideoCapture('fire.mp4')
model = YOLO('best.pt')
classnames = ['smoke']

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(1200,800))
    result = model(frame,stream= True)
    
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 30:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
                # x1,y1,x2,y2 = (int(i) for i in box.xyxy[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2), (0,0,255),5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1+8,y1 + 100],scale = 1.5, thickness=2)
                
    cv2.imshow('frame',frame)
    cv2.waitKey(1)
            