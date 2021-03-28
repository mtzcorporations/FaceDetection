import cv2
import time
import dlib
from rsize import resizePic

re=resizePic
# Load the detector
detector = dlib.get_frontal_face_detector()
# read the image
avg=0
#img = cv2.imread("slika1.jpg")
for i in range (0,20):
    cap = cv2.VideoCapture('video2.mp4')
    len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(len_frames)
    count=1

    while True:
            _, img = cap.read()  #get da FRAME 
            # Convert image into grayscale
            #gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
            start = time.time()
            # Use detector to find landmarks
            faces = detector(img,0)
            end = time.time()
            # print(start-end)
            avg=avg+(end-start)
            for face in faces:
                x1 = face.left() # left point
                y1 = face.top() # top point
                x2 = face.right() # right point
                y2 = face.bottom() # bottom point
                # Draw a rectangle
                cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=4)
            resize = re.ResizeWithAspectRatio(img, width=800) # Resize by width OR
            resize = re.ResizeWithAspectRatio(img, height=800)
            cv2.imshow('resize', resize)
            # Stop if escape key is pressed
            k = cv2.waitKey(30) & 0xff
            if len_frames==count:
                break
            count+=1
            if k==27:
               break
# Release the VideoCapture object
print("Execution Time (in seconds) :",avg/20, "per frame AVG ",avg/(len_frames*20))
cap.release()
