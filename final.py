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

def pinch_distance(point1, point2):
    return math.hypot(point1.x - point2.x, point1.y - point2.y)

def pinching(hand_landmarks, threshold=0.05):
    return pinch_distance(hand_landmarks[4], hand_landmarks[8]) < threshold

def finger_up_down(hand_landmarks, tip_ldm, pip_ldm):
    return hand_landmarks[tip_ldm].y < hand_landmarks[pip_ldm].y

def finger_status(hand_landmarks):
    return [
        finger_up_down(hand_landmarks, 8, 6),
        finger_up_down(hand_landmarks, 12, 10),
        finger_up_down(hand_landmarks, 16, 14),
        finger_up_down(hand_landmarks, 20, 18)
    ]

def elapsed_time(timestamp):
    if len(timestamp) < 2:
        return 0
    return timestamp[-1] - timestamp[0]

def swipe(positions, time_t):
    dt = elapsed_time(time_t)
    if len(positions) < 15 or dt == 0:
        return None
        
    threshold = 100
    start_x, start_y = positions[0]
    end_x, end_y = positions[-1]

    dx = end_x - start_x
    dy = end_y - start_y

    if abs(dx) > abs(dy):
        velocity_x = dx / dt
        if dx > threshold and abs(velocity_x) >= 200:
            return 'right'
        elif dx < -threshold and abs(velocity_x) >= 200:
            return 'left'
    else:
        velocity_y = dy / dt
        if dy > threshold and abs(velocity_y) >= 200:
            return 'down'
        elif dy < -threshold and abs(velocity_y) >= 200:
            return 'top'
            
    return None

def drone_worker():
    while True:
        command = command_queue.get()
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
        elif command == "left":
            drone.move_left(30)
        elif command == "right":
            drone.move_right(30)
        command_queue.task_done()

# -----------------------------VARIABLES-------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17), (0, 9), (0, 13)
]

positions = deque(maxlen=15)
time_t = deque(maxlen=15)

# Limit max queue size to 1 so commands don't buffer up lag
command_queue = Queue(maxsize=1)

drone = Drone()
drone.connect()

last_command_time = 0
debounce_time = 0.5

drone_thread = Thread(target=drone_worker, daemon=True)
drone_thread.start()

# -----------------------------------------------------------------
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
# Optional: Force camera resolution to 640x480 for higher FPS
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Standardized millisecond timestamp
    frame_timestamp_ms = int(time.time() * 1000)
    detector.detect_async(mp_image, frame_timestamp_ms)

    if latest_result and latest_result.hand_landmarks:
        h, w, _ = frame.shape
        now = time.time()
       
        for hand_landmarks in latest_result.hand_landmarks:
            # Draw points
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # Track wrist position and timestamp
            wrist = hand_landmarks[0]
            wrist_idx = (int(wrist.x * w), int(wrist.y * h))
            positions.append(wrist_idx)
            time_t.append(now)

            cv2.putText(frame, str(wrist_idx), wrist_idx, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 36, 0), 2)

            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                start_lm = hand_landmarks[start_idx]
                end_lm = hand_landmarks[end_idx]
                pt1 = (int(start_lm.x * w), int(start_lm.y * h))
                pt2 = (int(end_lm.x * w), int(end_lm.y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # Detect Gestures
            finger_states = finger_status(hand_landmarks)
            direction = swipe(positions, time_t)

            if (now - last_command_time) >= debounce_time:
                command_to_send = None

                if direction == 'top':
                    command_to_send = "take_off"
                elif direction == 'down':
                    command_to_send = "land"
                elif finger_states == [1, 0, 0, 0]:
                    command_to_send = "up"
                elif finger_states == [1, 1, 0, 0]:
                    command_to_send = "down"
                elif finger_states == [1, 0, 0, 1]:
                    command_to_send = "forward"
                elif finger_states == [0, 1, 1, 0]:
                    command_to_send = "backward"
                elif finger_states == [1, 1, 1, 0]:
                    command_to_send = "left"
                elif finger_states == [1, 1, 1, 1]:
                    command_to_send = "right"

                if command_to_send:
                    # Drop old commands if drone is slow to prevent queuing latency
                    if command_queue.full():
                        try:
                            command_queue.get_nowait()
                        except:
                            pass
                    command_queue.put(command_to_send)
                    last_command_time = now

                if direction is not None:
                    positions.clear()
                    time_t.clear()

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cap.release()
cv2.destroyAllWindows()