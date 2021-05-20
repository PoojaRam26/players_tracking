import numpy as np 
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from numpy import argmax
import time
import imutils

modelss = load_model(r'model\model.h5')


cap = cv2.VideoCapture('video.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


fourcc = cv2.VideoWriter_fourcc('m','p','4','v')


out = cv2.VideoWriter('final_out.avi',fourcc,fps, (frame_width,frame_height))



if (cap.isOpened()== False): 
    print("Error opening video stream or file")
    

net = cv2.dnn.readNet("model\yolov3.weights", "model\yolov3.cfg")


classes = []
with open("model\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3)) 




def get_players(outs,height, width):
    class_ids = []
    confidences = []
    boxes = []
    players=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label=='person':
                players.append(boxes[i])
            
    return players



opr=0
while(cap.isOpened()):
    ret, frame = cap.read()
    
    players=[]

    if opr<310:
        opr=opr+1
        continue
    
    if ret == True :
  
        copy=frame.copy()
        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        height, width, channels = frame.shape
        
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        start = time.time()
        outs = net.forward(output_layers)
        end = time.time()
        outs=get_players(outs, height, width)
        
        for i in range(len(outs)):
            x, y, w, h = outs[i]
            roi = frame[y:y+h,x:x+w]
            
            try:
                roi=cv2.resize(roi, (96,96))
            except:
                continue
            ym=modelss.predict(np.reshape(roi,(1,96,96,3)))
            ym=argmax(ym)
            
            players.append([x,y,w,h,ym])
            
            if ym==0:
                cv2.rectangle(copy, (x, y), (x + w, y + h), (255,0,), 2)
            elif ym==1:
                cv2.rectangle(copy, (x, y), (x + w, y + h), (0,255,0), 2)
            elif ym==2:
                cv2.rectangle(copy, (x, y), (x + w, y + h), (0,0,255), 2)
            
        out.write(copy)       
        cv2.imshow('final_output_video',copy)

        if cv2.waitKey(1) == 27:
            break

    

cap.release()
out.release()
cv2.destroyAllWindows()

