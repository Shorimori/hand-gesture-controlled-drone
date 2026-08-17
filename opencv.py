import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

latest_result = None

def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# Initialize MediaPipe Tasks Hand Landmarker (New API)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_result
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0
drawing = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Send frame asynchronously to the new MediaPipe Tasks API
    frame_timestamp_ms = int(time.time() * 1000)
    detector.detect_async(mp_image, frame_timestamp_ms)

    if latest_result and latest_result.hand_landmarks:
        for hand_landmarks in latest_result.hand_landmarks:
            # Draw visual landmarks on the frame
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            # Check if index finger is up (Tip y < PIP y)
            index_tip = hand_landmarks[8]
            index_pip = hand_landmarks[6]

            if index_tip.y < index_pip.y:
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                if not drawing:
                    prev_x, prev_y = cx, cy
                    drawing = True
                else:
                    cv2.line(canvas, (prev_x, prev_y), (cx, cy),(94, 172, 203), 5) # Blue line, thickness 5
                    prev_x, prev_y = cx, cy
            else:
                drawing = False

    # Blend the drawing canvas with the live video frame
    frame = cv2.add(frame, canvas)

    cv2.imshow("Screen Drawing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        break
    elif key == ord('r'):  # Press 'r' to clear the screen
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()