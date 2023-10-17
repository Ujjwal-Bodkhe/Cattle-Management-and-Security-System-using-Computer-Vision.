from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import time
from pygame import mixer
def Sound():
    # playsound('Alarm.mp3')
    mixer.init()
    mixer.music.load("../Alarm.mp3")
    mixer.music.play()
    while mixer.music.get_busy():  # wait for music to finish playing
        time.sleep(1)

cap = cv2.VideoCapture(0) #For WebCam
cap.set(3,1280)
cap.set(4,720)
#cap = cv2.VideoCapture("../Cows1.mp4") # For Video
language = "en"


model = YOLO("../System.pt")

classNames = ['Dog', 'Koala', 'White-beaked dolphin', 'Zebra', 'a-pig', 'antelope', 'badger', 'bat', 'bear',
              'bison', 'boar', 'cat', 'chimpanzee', 'cow', 'coyote', 'crab', 'crocodie', 'crow', 'deer', 'dolphin',
              'donkey', 'duck', 'eagle', 'elephant', 'flamingo', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'hamster',
              'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'human', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo',
              'killer-whale', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi',
              'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'peguin', 'pelecaniformes', 'pigeon', 'porcupine', 'possum',
              'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel',
              'starfish', 'swain', 'tiger', 'turkey', 'turtle', 'undetected', 'whale', 'whale-shark', 'wolf', 'wombat', 'woodpecker']

#mask = cv2.imread("../mask.png")


#Tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limits = [400,400,700,700]
limitT = [400,400,700,700]
totalCount=[]

while True:
    success, img = cap.read()
    #imgRegion = cv2.bitwise_and(img,mask)

    imgGraphics = cv2.imread("../AdobeStock_258946740_Preview.png",cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,imgGraphics,(0,0))
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
         #Confidance
         conf = math.ceil((box.conf[0]*100))/100

        #class Name
         cls = int(box.cls[0])

         currentClass = classNames[cls]
         if currentClass == "tiger" or currentClass == "leopard" or currentClass == "lion":
             Sound()

         elif currentClass=="cow" or currentClass=="reindeer" or currentClass=="bison" or currentClass=="ox" and conf>0.3:

          # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),scale=0.7,thickness=1,offset=5)
          # cvzone.cornerRect(img, (x1, y1, w, h), l=10,rt = 5)

          currentArray = np.array([x1,y1,x2,y2,conf])
          detections = np.vstack((detections,currentArray))
    resultsTracker = tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    cv2.line(img, (limitT[0], limitT[1]), (limitT[2], limitT[3]), (0, 0, 255), 5)


    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)

        w,h = x2-x1,y2-y1
        cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=1.5, thickness=2, offset=5)

        cx,cy = x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limits[0]<cx<limits[2] and limits[1]-5<cy<limits[1]+5:
            if totalCount.count(id)==0:
              totalCount.append(id)
              cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
              #Sound()





    # cvzone.putTextRect(img, f'count:{len(totalCount)}', (50,50))
    cv2.putText(img,str(len(totalCount)),(200,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
    cv2.imshow("Image",img)
    #cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
