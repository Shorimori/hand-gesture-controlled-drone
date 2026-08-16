import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math

latest_result = None
def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result 
    if result.hand_landmarks:
        print('hand detected', len(result.hand_landmarks))
    else:
        print('error')

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

base_options = python.BaseOptions(model_asset_path = 'hand_landmarker.task')

options = vision.HandLandmarkerOptions(
            base_options = base_options,
            num_hands =2,
            min_hand_detection_confidence= 0.5,
            running_mode = vision.RunningMode.LIVE_STREAM,
            result_callback = print_result
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

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

        for hand_landmarks in latest_result.hand_landmarks:
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx,cy), 4,(0,255,0), -1)
        
        finger_states = finger_status(hand_landmarks)
        if finger_states ==[0,0,1,1]:
            cv2.putText(frame,'peace', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF==ord('c'):
        break


cap.release()
cv2.destroyAllWindows()