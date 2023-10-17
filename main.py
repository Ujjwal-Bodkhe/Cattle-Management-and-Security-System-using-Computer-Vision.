from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import time
from pygame import mixer

# cap = cv2.VideoCapture(0) #For WebCam
# cap.set(3,1280)
# cap.set(4,720)
cap = cv2.VideoCapture("Cows in a row.mp4") # For Video


model = YOLO("../System.pt")

classNames = ['Dog', 'Koala', 'White-beaked dolphin', 'Zebra', 'a-pig', 'antelope', 'badger', 'bat', 'bear',
              'bison', 'boar', 'cat', 'chimpanzee', 'cow', 'coyote', 'crab', 'crocodie', 'crow', 'deer', 'dolphin',
              'donkey', 'duck', 'eagle', 'elephant', 'flamingo', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'hamster',
              'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'human', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo',
              'killer-whale', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi',
              'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'peguin', 'pelecaniformes', 'pigeon', 'porcupine', 'possum',
              'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel',
              'starfish', 'swain', 'tiger', 'turkey', 'turtle', 'undetected', 'whale', 'whale-shark', 'wolf', 'wombat', 'woodpecker']


#Tracker
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
limits = [700,0,700,720]
totalCount=[]


while True:
    success, img = cap.read()
    results = model(img,stream=True)
    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
        #bounding box
         x1,y1,x2,y2 = box.xyxy[0]
         x1, y1, x2, y2 = int(x1), int(y1), int(x2),int(y2)
         #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

         w, h = x2-x1,y2-y1
         # cvzone.cornerRect(img,(x1,y1,w,h))
         #Confidance
         conf = math.ceil((box.conf[0]*100))/100

        #class Name
         cls = int(box.cls[0])
         currentClass = classNames[cls]

         if currentClass=="cow" or currentClass=="ox" or currentClass=="reindeer" and conf > 0.2:
           currentArray = np.array([x1, y1, x2, y2, conf])
           detections=np.vstack((detections,currentArray))
    resultsTracker = tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)


    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        print(result)

        w,h = x2-x1,y2-y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=1.5, thickness=2, offset=5)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0]<cx<limits[2] and limits[1]+40<cy<limits[3]-40:
            if totalCount.count(id)==0:
              totalCount.append(id)
              cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)


        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),scale=1,thickness=2)

    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
