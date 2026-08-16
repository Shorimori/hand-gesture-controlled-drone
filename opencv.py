import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

#to capture live stream
cap = cv2.VideoCapture(0)
frame_count = 0


#to continue capturing the frame
while cap.isOpened():
    success, frame = cap.read()
    height,width,z = frame.shape
    if not success:
        continue
    frame_count += 1

    cv2.putText(frame,f"Frame: {frame_count}",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)
    frame = cv2.flip(frame,1)
    cv2.line(frame, (0,height//2), (width,height//2),(0,255,0), 3)
    cv2.line(frame,(width//2,height),(width//2,0),(255,255,255),4)
    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF==ord('q'):
        break




    
#cleanup after loops to close all windows
cap.release()
cv2.destroyAllWindows()

