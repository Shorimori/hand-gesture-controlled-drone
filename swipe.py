import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import math
from collections import deque

latest_result = None
def print_result(result,output_image, timestamp):
    global latest_result
    latest_result = result
    if result.hand_landmarks:
        print('hands detected:',len(result.hand_landmarks))
    else:
        print('no hand detected')

def swipe(positions):
    threshhold= 100
    start_x, start_y = positions[0]
    end_x, end_y = positions[-1]

    dx= end_x-start_x
    dy=end_y - start_y
  
    print('dx:', dx, 'dy:',dy)
   

    if abs(dx)>abs(dy):
        dy=0
    else:
        dx=0
    velocity_x= dx/elapsed_time(time_t)
    velocity_y=dy/elapsed_time(time_t)
    print('velocity_x:',velocity_x)
    print('velocity_y:',velocity_y)
    if dx>threshhold and abs(velocity_x)>=800:
        print('right')
    elif dx<-threshhold and abs(velocity_x)>=800:
        print('left')

    elif dy>threshhold and abs(velocity_y)>=800:
        print('bottom')

    elif dy<-threshhold and abs(velocity_y)>=800:
        print('top')
    else:
        print('slow!!!!!')
    

def elapsed_time(timestamp):
    return(timestamp[-1]-timestamp[0])





HAND_CONNECTIONS = [ (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), 
    (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),(0,9),(0,13)]
positions =deque(maxlen=15)
time_t =deque(maxlen=15)
# threshhold = 10
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

options = vision.HandLandmarkerOptions(
    base_options= base_options,
    num_hands =1,
    min_hand_detection_confidence=0.5,
    running_mode = vision.RunningMode.LIVE_STREAM,
    result_callback= print_result
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data= rgb_frame)
    time_stamp = int(cv2.getTickCount()/cv2.getTickFrequency()*1000)
    detector.detect_async(mp_image, time_stamp)

    if latest_result and latest_result.hand_landmarks:
        h,w,z = frame.shape
        start_time = time.time()
        for hand_landmarks in latest_result.hand_landmarks:
            for lm in hand_landmarks:
                cx, cy = int(lm.x*w),int(lm.y*h)
                cv2.circle(frame, (cx,cy),4,(180, 32, 46),-1)
            wrist = hand_landmarks[0]
            wrist_idx = (int(wrist.x * w), int(wrist.y * h))
            positions.append(wrist_idx)
            end_time = time.time()
            time_t.append(end_time)


                

            cv2.putText(frame, str(wrist_idx), wrist_idx, cv2.FONT_HERSHEY_SIMPLEX,2,(255,36,0), 2)
            

            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y *h))
                end = (int(hand_landmarks[end_idx].x*w), int(hand_landmarks[end_idx].y *h))

                cv2.line(frame, start, end,(223, 32, 43),2)
            

            
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cap.release()
cv2.destroyAllWindows()



print(positions)
print(elapsed_time)
elapsed_time(time_t)
swipe(positions)
print('no of positions:', len(positions))
print('elapsed_time:',elapsed_time(time_t))