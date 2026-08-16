from pysimverse import Drone
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from collections import deque


latest_result = None
def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result 
    if result.hand_landmarks:
        print('hand detected', len(result.hand_landmarks))
    else:
        print('error')

def pinch_distance(point1, point2):
    distance = math.sqrt((point1.x - point2.x)**2 + (point1.y- point2.y)**2)
    return distance

def pinching(hand_landmarks, threshhold= 0.05):
    thumb = hand_landmarks[4]
    index = hand_landmarks[8]
    distance = pinch_distance(thumb, index)
    return distance< threshhold

def finger_up_down(hand_landmarks,tip_ldm, pip_ldm):
    tip_y = hand_landmarks[tip_ldm].y
    pip_y = hand_landmarks[pip_ldm].y
    return tip_y > pip_y       #if true, finger is down


def finger_status(hand_landmarks):
    states = []
    states.append(finger_up_down(hand_landmarks,8,6))
    states.append(finger_up_down(hand_landmarks,12,10))
    states.append(finger_up_down(hand_landmarks,16,14))
    states.append(finger_up_down(hand_landmarks,20,18))
    return states

def finger_up_down(hand_landmarks,tip_ldm, pip_ldm):
    tip_y = hand_landmarks[tip_ldm].y
    pip_y = hand_landmarks[pip_ldm].y
    return tip_y > pip_y       #if true, finger is down

def gesture():
    if finger_status(hand_landmarks):
        return True
    else:
        return False
            
    
frame_count =1
previous_state = None
HAND_CONNECTIONS = [ (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),(0,9),(0,13)]
drone = Drone()
drone.connect()
base_options = python.BaseOptions(model_asset_path = 'hand_landmarker.task')

options = vision.HandLandmarkerOptions(
            base_options = base_options,
            num_hands =1,
            min_hand_detection_confidence= 0.5,
            running_mode = vision.RunningMode.LIVE_STREAM,
            result_callback = print_result
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
drone.take_off()
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        continue
    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data= rgb_frame)
    time_stamp = int(cv2.getTickCount()/cv2.getTickFrequency()*1000)
    detector.detect_async(mp_image, time_stamp)

    if latest_result and latest_result.hand_landmarks:
        h,w,_ = frame.shape

        for hand_landmarks in latest_result.hand_landmarks:
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx,cy), 4,(0,255,0), -1)

        
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_lm = hand_landmarks[start_idx]
            end_lm = hand_landmarks[end_idx]

            start_point_x, start_point_y = int(start_lm.x * w), int(start_lm.y * h)
            end_point_x, end_point_y = int(end_lm.x * w), int(end_lm.y * h)

            cv2.line(frame, (start_point_x, start_point_y),(end_point_x, end_point_y),(0,255,0), 3)

        if frame_count%5==0:
            finger_states = finger_status(hand_landmarks)
            if gesture()==True:   
                if finger_states ==[0,1,1,1]:
                    drone.move_up(30)
                elif finger_states ==[0,0,1,1]:
                    drone.move_down(30)
                elif finger_states ==[0,0,0,1]:
                    drone.move_forward(30)
                elif finger_states ==[0,0,0,0]:
                    drone.land()
        frame_count +=1
        if frame_count ==16:
            frame_count=1

       
        
        
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF==ord('c'):
        break


cap.release()
cv2.destroyAllWindows()





# if pinching(hand_landmarks):
        #     drone.take_off()


