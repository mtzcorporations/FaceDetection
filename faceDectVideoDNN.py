import cv2
import dlib
import numpy as np
from rsize import resizePic

re=resizePic
# Load the detector
detector =  cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
# read the image
#img = cv2.imread("slika1.jpg")
cap = cv2.VideoCapture('video.mp4')
while True:
        _, img = cap.read()  #get da FRAME 
        # Convert image into grayscale
        #gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

        # Use detector to find landmarks
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        detector.setInput(blob)
        faces= detector.forward()
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
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
           break
# Release the VideoCapture object
cap.release()
