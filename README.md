# Hand Gesture Controlled Drone

Flying a drone with hand gestures. Live webcam footage is passed through MediaPipe's `HandLandmarker` to get 21 hand landmarks, simple geometric rules turn those into named gestures, and the gestures drive a drone over MAVLink — running in ArduPilot SITL + Gazebo. Can be shifted to hardware later as it has the same interface.

## How it works
```
webcam -> HandLandmarker -> gesture classifier -> debounce -> MAVLink velocity setpoints -> ArduPilot (SITL or real FC)
```

- **Gesture recognition** is rule-based. Each of the 21 landmarks per hand is used directly — finger extension is read from distance-to-wrist, the thumb from distance to the far palm corner — and a lookup table maps finger patterns to gesture names. No training data, no ML dependencies beyond MediaPipe itself.
- **Hand tilt** doubles as a yaw modifier, layered on top of whatever gesture is active — tilt while hovering turns in place, tilt while flying forward turns while moving.
- **Movement is velocity-based**, not distance-based (`SET_POSITION_TARGET_LOCAL_NED` in `MAV_FRAME_BODY_NED`). The drone keeps moving in a direction for as long as a gesture is held, and switches direction immediately when the gesture changes — no waiting for a discrete move to finish.
- **A background thread** keeps the drone's setpoint stream alive at 20Hz (required — ArduPilot stops the vehicle if setpoints go quiet for ~3s) while the camera loop stays completely non-blocking.

## Gesture map

| Gesture | Fingers extended | Command |
|---|---|---|
| THUMB_OUT | thumb | takeoff |
| FIST | none | land |
| OPEN_PALM | all five | hover |
| PEACE | index, middle | forward |
| THREE | index, middle, ring | back |
| L_SHAPE | thumb, index | left |
| SHAKA | thumb, pinky | right |
| POINT | index | ascend |
| SPIDER_MAN | thumb, index, pinky | descend |

Tilting the hand left/right while any movement gesture is held adds yaw in that direction.

## Setup

### 1. Ubuntu 22.04 VM (VirtualBox)

ArduPilot SITL and Gazebo run on Linux. This project was built against Ubuntu 22.04 in a VirtualBox VM on a Windows host. Wanted to dual boot linux on my laptop but I was facing a lot of problems TvT....

### 2. ArduPilot SITL + Gazebo

Follow the [ArduPilot Gazebo plugin setup](https://github.com/ArduPilot/ardupilot_gazebo) to build `ardupilot_gazebo` from source. Once installed:

```bash
# terminal 1 - Gazebo
gz sim -v4 -r <your_world>.sdf

# terminal 2 - SITL
cd ~/ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter --console --map --out=udp:127.0.0.1:14551
```

Wait for SITL to report `pre-arm good` before running anything else - arming will fail otherwise.

### 3. Python dependencies

```bash
pip install mediapipe opencv-python pymavlink --break-system-packages
```

### 4. Download the hand landmark model

Not tracked in this repo - download it into the project root:

```bash
wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 5. Set the connection string

In `connect.py`:

```python
CONNECTION = "udp:127.0.0.1:14550"   # or 14551 if using the Mission Planner mirror port
```

If the camera and SITL are on different machines (e.g. camera on a Windows host, SITL in the VM), point this at the VM's IP instead of `127.0.0.1`.

### 6. Run

```bash
python3 connect.py         # check: arms, takes off, flies a short pattern, lands
python3 gesture_drone.py   # the real thing
```

Press **ESC** to quit - this lands the drone on the way out.

## Project structure

| File | Purpose |
|---|---|
| `gesture_drone.py` | Webcam capture, landmark detection, gesture classification, debouncing, and the camera-loop side of the app. |
| `connect.py` | All MAVLink/ArduPilot knowledge lives here. Nothing else in the project imports pymavlink, so the gesture code stays independent of the flight stack. |
| `hand_landmarker.task` | MediaPipe hand landmark model (downloaded, not committed - see Setup). |

## Notes

- **Why velocity control, not distance moves.** An earlier version issued fixed-distance commands (`move_forward(100)`); the drone would commit to completing that move before accepting anything else, which meant gestures were ignored mid-motion. Velocity setpoints solve this - a command is a persistent state, not a task to finish, so switching gestures changes the drone's behavior immediately.
- **Threading.** MAVLink calls like takeoff/land block for real seconds. A worker thread owns all drone communication so the camera feed never stalls; the camera loop and the worker share a single "currently wanted command" rather than a queue, so the drone always acts on the most recent gesture rather than working through a backlog of stale ones.
- **Model paths are resolved relative to the script's own location** (via `__file__`), not the terminal's working directory, so the project runs correctly regardless of where it's launched from.

## License

MIT (update as needed)
