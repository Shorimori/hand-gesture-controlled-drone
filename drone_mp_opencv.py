import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

#to capture live stream
cap = cv2.VideoCapture(0)

#to continue capturing the frame
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

#cleanup after loops to close all windows
cap.release()
cv2.destroyAllWindows()

#converting BRG to RGB
rgb_frame= cv2.cvtCOLOR(frame, cv2. COLOR_BRG2RGB)

#converting to Mediapipe format of images
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data= rgb_frame)


# to extract the particular model 
base_options = python.BaseOptions(model_asset_path = 'hand_landmarker.task')

#creating callback 
def print_result(result, output_image, timestamp_ms):
    if result.hand_landmarks:
        print('detected hands:', len(result.hand_landmarks))
    else:
        print('error')

# to extract the task related options
options = vision.HandLandmarkerOptions(
    base_options= base_options,
    num_hands=2,
    min_hand_detection_confidence= 0.5,
    running_mode = vision.RunningMode.LIVE_STREAM,
    result_callback = print_result
)

detector = vision.HandLandmarker.create_from_options(options)

timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
detector.detect_async(mp_image, timestamp_ms)
