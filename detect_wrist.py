import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import math


latest_result = None
def print_result(result,output_image, timestamp):
    global latest_result
    latest_result = result
    if result.hand_landmarks:
        print('hands detected:',len(result.hand_landmarks))
    else:
        print('no hand detected')


HAND_CONNECTIONS = [ (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), 
    (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),(0,9),(0,13)]
positions =[]
threshhold = 10
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

options = vision.HandLandmarkerOptions(
    base_options= base_options,
    num_hands =4,
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

    if latest_result and latest_result.hand_landmarks and latest_result.handedness:
        h,w,z = frame.shape
        for i, hand_landmarks in enumerate(latest_result.hand_landmarks):

            hand_name = latest_result.handedness[i][0].category_name
            if hand_name == "Left":
                hand_name = "Right"
            else:
                hand_name = "Left"

            # Get wrist position
            wrist = hand_landmarks[0]
            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)

            cv2.putText(
            frame,
            f"{hand_name} Hand",
            (wrist_x, wrist_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
            )

            # Draw landmarks
            for lm in hand_landmarks:
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                cv2.circle(frame,(cx, cy),4,(180, 32, 46),-1)
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


        

        