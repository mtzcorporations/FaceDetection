import cv2
import time
from rsize import resizePic

re=resizePic
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
avg=0
# To capture video from webcam. 
#cap = cv2.VideoCapture(0)
# To use a video file as input 
for i in range (0,20):
    cap = cv2.VideoCapture('distance.mp4')
    len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count=1
    while True:
        # Read the frame
        _, img = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces=0
        start = time.time()
       # for i in range (0,100):
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        end = time.time()
       # print(start-end)
        avg=avg+(end-start)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display
        
        resize = re.ResizeWithAspectRatio(img, width=800) # Resize by width OR
        resize = re.ResizeWithAspectRatio(img, height=800) # Resize by height 

        cv2.imshow('resize', resize)
        if len_frames==count:
            break
        count+=1
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break

print("Execution Time (in seconds) :",avg/20, "per frame AVG ",avg/(20*len_frames))
# Release the VideoCapture object
cap.release()
