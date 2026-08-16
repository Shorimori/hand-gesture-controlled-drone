import mediapipe as mp
from mediapipe.tasks import python #to point to model files
from mediapipe.tasks.python import vision #contains task classes like HandLandmarker, HandLandmarkerOptions

def print_result(result, output_result, time_stamp):
    global latest_result
    latest_result= result,
    if result.hand_landmarks:
        print('hand detected:', len(result.hand_landmarks))
    else:
        print('no hand detected')


# to extract the particular model 
base_options = python.BaseOptions(model_asset_path = 'hand_landmarker.task')

# to extract the task related options
options = vision.HandLandmarkerOptions(
    base_options= base_options,
    num_hands=2,
    min_hand_detection_confidence= 0.5,
    running_mode = vision.RunningMode.LIVE_STREAM,
    result_callback = print_result
)

detector = vision.HandLandmarker.create_from_options(options)

#image input
image = mp.Image.create_from_file('test_hand.jpg')

# detect the hand
detection_result= detector.detect(image)
print(detection_result.hand_landmarks)

wrist = detection_result.hand_landmarks[0][0]
print(wrist.x, wrist.y, wrist.z)





