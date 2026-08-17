import cv2
import math
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pysimverse as psv
from pysimverse import Drone

# --- CONFIGURATION ---

SPEED = 40       
STEP_TIME = 0.25  # Duration (seconds) per path segment
MIN_DIST = 15     # Minimum pixel distance between recorded points

#----------------------------------------------
base_options = python.BaseOptions(model_asset_path= 'hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)


def get_pointing_pos(frame, timestamp_ms):
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # Synchronous frame detection (no async callbacks required)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.hand_landmarks:
        return None

    lm = result.hand_landmarks[0]
    h, w, _ = frame.shape

    if (lm[8].y < lm[6].y) and (lm[12].y > lm[10].y):
        return int(lm[8].x * w), int(lm[8].y * h)
    return None


def fly_path(points):
    """Executes RC speed vector controls to trace the drawn pixel points."""
    if len(points) < 2:
        print("Draw a path first!")
        return

    # Downsample path (take every 3rd point) for smooth drone motion
    waypoints = points[::3]
    if waypoints[-1] != points[-1]:
        waypoints.append(points[-1])

    print(f"Flying path with {len(waypoints)} waypoints...")

    drone = Drone()
    drone.connect()
    drone.take_off()

    for i in range(1, len(waypoints)):
        x1, y1 = waypoints[i - 1]
        x2, y2 = waypoints[i]

        dx = x2 - x1
        dy = -(y2 - y1)  # Invert Y: Screen Y increases downward, altitude increases upward

        length = math.hypot(dx, dy)
        if length == 0:
            continue

        # Convert segment direction to proportional RC speed
        lr = int((dx / length) * SPEED)  # Left / Right
        ud = int((dy / length) * SPEED)  # Up / Down

        drone.send_rc_control(lr, 0, ud, 0)
        time.sleep(STEP_TIME)

    # Stop and land
    drone.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)
    drone.land()
    drone.shutdown()
    print("Flight complete!")


def main():
    cap = cv2.VideoCapture(0)
    path = []

    print("Controls: Point index finger to draw | 'e' = execute | 'c' = clear | 'q' = quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        timestamp_ms = int(time.time() * 1000)

        pos = get_pointing_pos(frame, timestamp_ms)

        # Store landmark position when pointing gesture is active
        if pos:
            if not path or math.hypot(pos[0] - path[-1][0], pos[1] - path[-1][1]) > MIN_DIST:
                path.append(pos)

        # Draw recorded lines
        for i in range(1, len(path)):
            cv2.line(frame, path[i - 1], path[i], (0, 255, 0), 3)

        status = "DRAWING" if pos else "IDLE"
        cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Air Draw & Fly", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            path.clear()
        elif key == ord('e'):
            fly_path(path)
            path.clear()

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()