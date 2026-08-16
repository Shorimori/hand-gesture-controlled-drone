# has functions: print_result, pinch_distance, pinching, finger_up_down, finger_status, swipe, elapsed_time, gesture
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from collections import deque

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
    return tip_y < pip_y       #if true, finger is up


def finger_status(hand_landmarks):
    states = []
    states.append(finger_up_down(hand_landmarks,8,6))
    states.append(finger_up_down(hand_landmarks,12,10))
    states.append(finger_up_down(hand_landmarks,16,14))
    states.append(finger_up_down(hand_landmarks,20,18))
    return states

def swipe(positions):
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
        dir = 'bottom'

    elif dy<-threshhold and abs(velocity_y)>=200:
        dir='top'
    return dir
    
def elapsed_time(timestamp):
    return(timestamp[-1]-timestamp[0])


def gesture():
    if finger_status(hand_landmarks):
        return True
    else:
        return False
            