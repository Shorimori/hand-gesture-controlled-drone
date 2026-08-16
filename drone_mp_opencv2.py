import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---- Step 11: Callback function (must be defined BEFORE options) ----
latest_result = None

def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result
    if result.hand_landmarks:
        print(f"Detected {len(result.hand_landmarks)} hand(s)")
    else:
        print("No hand detected")

# ---- Step 2 & 3: BaseOptions + HandLandmarkerOptions ----
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_result
)

# ---- Step 4: Create the detector ----
detector = vision.HandLandmarker.create_from_options(options)

# ---- Step 8: Open webcam ----
cap = cv2.VideoCapture(0)

# ---- Step 9-13: Main loop ----
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Step 9 (cont.): BGR -> RGB conversion
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Step 12: Wrap as mp.Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Step 13: Timestamp + async detect
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    detector.detect_async(mp_image, timestamp_ms)

    # --- Draw landmarks on frame using latest available result ---
    if latest_result and latest_result.hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in latest_result.hand_landmarks:
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()