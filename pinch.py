import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math

latest_result = None

def print_result(result, output_image, timestamp_mp):
    global latest_result
    
    latest_result = result
    if result.hand_landmarks:
        print('Hands detected:', len(result.hand_landmarks))
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

base_options = python.BaseOptions(model_asset_path= 'hand_landmarker.task')

options= vision.HandLandmarkerOptions(
    base_options= base_options,
    num_hands = 4,
    min_hand_detection_confidence= 0.5,
    running_mode = vision.RunningMode.LIVE_STREAM,
    result_callback = print_result
)

detector = vision.HandLandmarker.create_from_options(options)
HAND_CONNECTIONS = [ (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), 
    (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),(0,9),(0,13)]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_img= mp.Image(image_format = mp.ImageFormat.SRGB, data= rgb_frame)

    time_stamp = int(cv2.getTickCount()/cv2.getTickFrequency()*1000)
    detector.detect_async(mp_img, time_stamp)

    if latest_result and latest_result.hand_landmarks:
        h, w, z = frame.shape
        for hand_landmarks in latest_result.hand_landmarks:
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (109, 255, 100), -1) #cv2.circle(image, (centre coordinates), radius, color RGB code, thickness how much filled)

            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_lm = hand_landmarks[start_idx]
                end_lm = hand_landmarks[end_idx]

                start_point = (int(start_lm.x * w), int(start_lm.y * h))
                end_point = (int(end_lm.x * w), int(end_lm.y * h))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        if pinching(hand_landmarks):
            cv2.putText(frame, "PINCH", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF==ord('c'):
        break


 
cap.release()
cv2.destroyAllWindows()




