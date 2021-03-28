import cv2
import dlib
import time
import numpy as np
from rsize import resizePic

re=resizePic
# Load the detector
detector =  cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
# read the image
#img = cv2.imread("slika1.jpg")

avg=0
for i in range (0,20):
    cap = cv2.VideoCapture('sideDist.mp4')
    len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    avg=0
    count=1
    while True:
            _, img = cap.read()  #get da FRAME 
            # Convert image into grayscale
            #gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
            start = time.time()
            # Use detector to find landmarks
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            detector.setInput(blob)
            faces= detector.forward()
            end = time.time()
            # print(start-end)
            avg=avg+(end-start)
            dims = img.shape
            h = dims[0]
            w = dims[1]
            for i in range(0, faces.shape[2]):
                confidence = faces[0, 0, i, 2]   
                if confidence>0.5:
                    box = faces[0,0,i,3:7] * np.array([w,h,w,h])
                    box = box.astype('int')
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    # Draw a rectangle
                    cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(25, 0, 255), thickness=4)
            resize = re.ResizeWithAspectRatio(img, width=800) # Resize by width OR
            resize = re.ResizeWithAspectRatio(img, height=800)
            cv2.imshow('resize', resize)
            if len_frames==count:
                break
            count+=1
            # Stop if escape key is pressed
            k = cv2.waitKey(30) & 0xff
            if k==27:
               break
print("Execution Time (in seconds) :",avg/20, "per frame AVG ",avg/(len_frames*20))
# Release the VideoCapture object
cap.release()
