from pysimverse import Drone
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from collections import deque
from queue import Queue
from threading import Thread




latest_result = None

# ------------------------------FUNCTIONS------------------------------
def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result 
    # if result.hand_landmarks:
    #     print('hand detected', len(result.hand_landmarks))
    # else:
    #     print('error')

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
    return tip_y < pip_y       #if true, finger is up


def finger_status(hand_landmarks):
    states = []
    states.append(finger_up_down(hand_landmarks,8,6))
    states.append(finger_up_down(hand_landmarks,12,10))
    states.append(finger_up_down(hand_landmarks,16,14))
    states.append(finger_up_down(hand_landmarks,20,18))
    return states

def swipe(positions, time_t):
    if len(positions)<15 or elapsed_time(time_t)==0:
        return None
    threshhold= 100
    start_x, start_y = positions[0]
    end_x, end_y = positions[-1]

    dx= end_x-start_x
    dy=end_y - start_y
    
   
    dir ='slow'

    if abs(dx)>abs(dy):
        dy=0
    else:
        dx=0
    velocity_x= dx/elapsed_time(time_t)
    velocity_y=dy/elapsed_time(time_t)
    
    if dx>threshhold and abs(velocity_x)>=200:
        dir = 'right'
    elif dx<-threshhold and abs(velocity_x)>=200:
        dir ='left'

    elif dy>threshhold and abs(velocity_y)>=200:
        dir = 'down'

    elif dy<-threshhold and abs(velocity_y)>=200:
        dir='top'
    return dir
    
def elapsed_time(timestamp):
    return(timestamp[-1]-timestamp[0])

def gesture(hand_landmarks, positions, time_t):
    finger_states = finger_status(hand_landmarks)
    swipe_direction = swipe(positions, time_t)

    if any(finger_states) or swipe_direction is not None:
        return True

    return False

def drone_worker():
    while True:
        command = command_queue.get()   # blocks THIS thread only, until something arrives
 
        if command == "take_off":
            drone.take_off()
        elif command == "land":
            drone.land()
        elif command == "up":
            drone.move_up(30)
        elif command == "down":
            drone.move_down(30)
        elif command == "forward":
            drone.move_forward(30)
        elif command == "backward":
            drone.move_backward(30)
 
        command_queue.task_done()
 
    
#-----------------------------VARIABLES-------------------------------


frame_f = deque(maxlen=16)  

HAND_CONNECTIONS = [ (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), 
    (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),(0,9),(0,13)]
positions = deque(maxlen=15)
time_t =deque(maxlen=16)
drone = Drone()
drone.connect()
command_queue= Queue()
last_command_time = 0
debounce_time = 0.5

drone_thread = Thread(target = drone_worker, daemon= True)
drone_thread.start()

#-----------------------------------------------------------------
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
frame_count = 0
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
        h,w,z = frame.shape
        start_time = time.time()

        for hand_landmarks in latest_result.hand_landmarks:
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx,cy), 4,(0,255,0), -1)
            wrist = hand_landmarks[0]
            wrist_idx = (int(wrist.x * w), int(wrist.y * h))
            positions.append(wrist_idx)
            end_time = time.time()
            time_t.append(end_time)
          
                

            cv2.putText(frame, str(wrist_idx), wrist_idx, cv2.FONT_HERSHEY_SIMPLEX,2,(255,36,0), 2)
        
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_lm = hand_landmarks[start_idx]
            end_lm = hand_landmarks[end_idx]

            start_point_x, start_point_y = int(start_lm.x * w), int(start_lm.y * h)
            end_point_x, end_point_y = int(end_lm.x * w), int(end_lm.y * h)

            cv2.line(frame, (start_point_x, start_point_y),(end_point_x, end_point_y),(0,255,0), 3)
        
        finger_states = finger_status(hand_landmarks)
        dir = swipe(positions, time_t)
        now = time.time()
        # if gesture(hand_landmarks, positions,time_t) and (now- last_command_time)>= debounce_time:
        if (now- last_command_time)>= debounce_time:

            command_to_send = None
            if dir == 'top':
                command_to_send = "take_off"
            elif finger_states == [0, 1, 1, 1]:
                command_to_send = "up"
            elif finger_states == [0, 0, 1, 1]:
                command_to_send = "down"
            elif finger_states == [0, 0, 0, 1]:
                command_to_send = "forward"
            elif finger_states == [0, 0, 0, 0]:
                command_to_send = "backward"
            elif dir == 'down':
                command_to_send = "land"
 
            if command_to_send is not None:
                command_queue.put(command_to_send)
                print("queue size:", command_queue.qsize())
                last_command_time = now
            if dir is not None:
                positions.clear()
                time_t.clear()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF==ord('c'):
        break


cap.release()
cv2.destroyAllWindows()