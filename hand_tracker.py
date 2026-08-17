"""
Gesture-controlled drone.

    GestureClassifier   landmarks -> gesture name
    DroneWorker          keeps the drone's velocity in sync with your hand
    LandmarkApp          camera loop, drawing, debouncing

Needs hand_landmarker.task and connect.py in the same folder.
ESC quits and lands the drone on the way out.
"""

import os
import time
import threading

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from connect import DroneLink

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

# which landmark joins to which, for drawing the skeleton
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# gesture -> command understood by DroneLink.send()
COMMANDS = {
    "THUMB_OUT":  "takeoff",
    "FIST":       "land",
    "OPEN_PALM":  "hover",
    "PEACE":      "forward",
    "THREE":      "back",
    "L_SHAPE":    "left",
    "SHAKA":      "right",
    "POINT":      "ascend",
    "SPIDER_MAN": "descend",
}

# Frames a new gesture must hold before it takes over. Hand poses flicker, and
# a hand moving between two gestures passes through others on the way.
STABLE_FRAMES = 4

# How often the active velocity is re-sent. RC control is a persistent setting
# that most firmware expects refreshed regularly, so this doubles as a
# keepalive - drop the resend and the drone may stall on its own.
SEND_INTERVAL = 0.05


class GestureClassifier:
    """21 landmarks in -> gesture name out. Knows nothing about cameras."""

    WRIST = 0
    THUMB_IP, THUMB_TIP = 3, 4
    PINKY_MCP = 17
    TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
    PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

    # (thumb, index, middle, ring, pinky) -> name
    PATTERNS = {
        (False, False, False, False, False): "FIST",
        (True,  True,  True,  True,  True):  "OPEN_PALM",
        (False, True,  False, False, False): "POINT",
        (True,  False, False, False, False): "THUMB_OUT",
        (False, True,  True,  False, False): "PEACE",
        (True,  True,  False, False, True):  "SPIDER_MAN",
        (False, True,  True,  True,  False): "THREE",
        (True,  True,  False, False, False): "L_SHAPE",
        (True,  False, False, False, True):  "SHAKA",
    }

    @staticmethod
    def _dist(a, b):
        # squared distance - every use is a comparison, so the square root
        # would change nothing except cost
        dx = a.x - b.x
        dy = a.y - b.y
        return dx * dx + dy * dy

    def finger_states(self, lm):
        """Which of the 5 fingers are extended right now? -> 5 booleans."""
        states = {
            # thumb splays sideways, so measure against the far palm corner
            "thumb": self._dist(lm[self.THUMB_TIP], lm[self.PINKY_MCP])
                     > self._dist(lm[self.THUMB_IP], lm[self.PINKY_MCP])
        }
        for name in self.TIPS:
            # extended = tip sits further from the wrist than the mid-joint
            states[name] = (self._dist(lm[self.WRIST], lm[self.TIPS[name]])
                            > self._dist(lm[self.WRIST], lm[self.PIPS[name]]))
        return states

    def classify(self, lm):
        s = self.finger_states(lm)
        key = (s["thumb"], s["index"], s["middle"], s["ring"], s["pinky"])
        return self.PATTERNS.get(key, "UNKNOWN")


class DroneWorker:
    """
    Keeps the drone doing whatever the current gesture says, on its own thread.

    Holds a single "currently wanted" command rather than a queue. A queue would
    let stale commands pile up so the drone acted on gestures from seconds ago;
    here a new gesture overwrites the old one and the drone always follows what
    your hand is doing now.

    Velocity commands return immediately, so the loop stays responsive.
    take_off and land genuinely take seconds - running them here is what keeps
    the camera feed from freezing during those.

    Every call is wrapped in try/except: without it, one failing call kills the
    thread silently and nothing is ever sent again.
    """

    def __init__(self, link):
        self.link = link
        self._wanted = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set(self, command):
        """Called from the camera loop. Returns instantly."""
        with self._lock:
            self._wanted = command

    def _loop(self):
        while self._running:
            with self._lock:
                command = self._wanted

            if command is None:
                time.sleep(0.02)
                continue

            try:
                self.link.send(command)
            except Exception as exc:
                print(f"[drone] '{command}' failed: {exc}")
                time.sleep(0.2)

            time.sleep(SEND_INTERVAL)

    def shutdown(self):
        with self._lock:
            self._wanted = None
        self._running = False
        self._thread.join(timeout=5)


class LandmarkApp:

    def __init__(self, source=0):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=2,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.classifier = GestureClassifier()

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError("Camera failed to open. Close other apps using "
                               "it, or try a different source index.")

        self.link = DroneLink()
        self.worker = DroneWorker(self.link)

        # debounce state
        self.candidate = None   # gesture currently being counted
        self.streak = 0         # consecutive frames it has held
        self.active = None      # gesture confirmed and being acted on

    def _handle_gesture(self, gesture):
        """
        A new gesture only takes over once it has held STABLE_FRAMES in a row,
        so a single flickered frame changes nothing. Once active it keeps being
        applied - the drone holds that velocity and keeps moving - until
        another gesture earns its frames and replaces it.

        When no hand is visible, fall back to hover rather than leaving the
        last velocity running.
        """
        if gesture == self.candidate:
            self.streak += 1
        else:
            self.candidate = gesture
            self.streak = 1

        if self.streak >= STABLE_FRAMES:
            self.active = gesture

        command = COMMANDS.get(self.active)
        if self.active is None:
            command = "hover"       # no hand - stop rather than drift
        self.worker.set(command)

    def draw(self, frame, lm, gesture):
        h, w = frame.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]

        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

        x, y = pts[GestureClassifier.WRIST]
        cv2.putText(frame, gesture, (x - 40, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def run(self):
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                result = self.detector.detect(image)

                gesture = None
                for i, lm in enumerate(result.hand_landmarks):
                    name = self.classifier.classify(lm)
                    self.draw(frame, lm, name)
                    if i == 0:
                        gesture = name   # only the first hand pilots

                self._handle_gesture(gesture)

                cv2.putText(frame, f"CMD: {COMMANDS.get(self.active, '-')}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2)

                cv2.imshow("Gestures", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            self.close()

    def close(self):
        self.worker.shutdown()
        try:
            self.link.send("land")   # safety net, no-op if already grounded
        except Exception as exc:
            print(f"[drone] landing failed: {exc}")
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    LandmarkApp().run()