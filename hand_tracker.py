"""
Video input + hand landmark detection + gesture classification. Simple version.

Two classes:
    GestureClassifier   landmarks -> gesture name   (pure logic, no camera/drawing)
    LandmarkApp          the loop + drawing

Needs hand_landmarker.task in the same folder. ESC to quit.
"""

import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# resolve the model path relative to this script, not the terminal's cwd
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
        (True, True, False, False, True)   : "SPIDER_MAN"
    }

    @staticmethod
    def _dist(a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

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


class LandmarkApp:

    def __init__(self, source=0):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=2,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.classifier = GestureClassifier()
        self.cap = cv2.VideoCapture(source)

    def draw(self, frame, lm, gesture):
        h, w, _ = frame.shape
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]

        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

        x, y = pts[GestureClassifier.WRIST]
        cv2.putText(frame, gesture, (x - 40, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def run(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = self.detector.detect(image)

            for lm in result.hand_landmarks:
                gesture = self.classifier.classify(lm)
                self.draw(frame, lm, gesture)

            cv2.imshow("Gestures", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    LandmarkApp().run()