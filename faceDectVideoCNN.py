import cv2
import dlib
import argparse
from rsize import resizePic

re=resizePic
# Load the detector
#detector = dlib.get_frontal_face_detector()
detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
# read the image
#img = cv2.imread("slika1.jpg")
cap = cv2.VideoCapture('video2.mp4')
while True:
        _, img = cap.read()  #get da FRAME 
        # Convert image into grayscale
        #gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

        # Use detector to find landmarks
        faces = detector(img,0)

        for face in faces:
            x1 = face.rect.left() # left point
            y1 = face.rect.top() # top point
            x2 = face.rect.right() # right point
            y2 = face.rect.bottom() # bottom point
            # Draw a rectangle
            cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=4)
        resize = re.ResizeWithAspectRatio(img, width=800) # Resize by width OR
        resize = re.ResizeWithAspectRatio(img, height=800)
        cv2.imshow('resize', resize)
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
           break
# Release the VideoCapture object
cap.release()
