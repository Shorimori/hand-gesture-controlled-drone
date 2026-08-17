"""
connect.py - PySimverse connection layer.

Uses velocity control (send_rc_control) rather than distance moves. The
difference matters: move_forward(100) is a commitment - the drone flies the
whole 100 units before it will look at anything else. send_rc_control sets a
velocity that simply persists until it is changed, so the drone keeps moving
while a gesture is held and switches direction the instant the gesture does.

Test the connection on its own, without the camera:

    python connect.py
"""

from pysimverse import Drone


# Velocity for each channel, roughly -100..100. Higher = faster.
SPEED = 50

# command -> (left_right, forward_backward, up_down, yaw)
# All zeros means hold position, which is what "hover" is.
VELOCITIES = {
    "hover":   (0, 0, 0, 0),
    "forward": (0, SPEED, 0, 0),
    "back":    (0, -SPEED, 0, 0),
    "left":    (-SPEED, 0, 0, 0),
    "right":   (SPEED, 0, 0, 0),
    "ascend":  (0, 0, SPEED, 0),
    "descend": (0, 0, -SPEED, 0),
}


class DroneLink:
    """
    Wraps PySimverse behind a single send(command) method.

    Movement is a persistent velocity, not a queued action, so send() never
    blocks and never needs to finish anything before the next command lands.
    """

    def __init__(self):
        self.drone = Drone()
        self.drone.connect()
        print("Drone connected.")

    @property
    def airborne(self):
        """Ask the drone, rather than tracking a flag that can drift."""
        try:
            return bool(self.drone.is_flying)
        except Exception:
            return False

    def send(self, command):
        """Command string -> drone action. Unknown commands are ignored."""
        if command == "takeoff":
            if not self.airborne:
                self.drone.take_off()
            return

        if command == "land":
            if self.airborne:
                self.stop()          # zero the velocities before landing
                self.drone.land()
            return

        if not self.airborne:
            return                   # velocity means nothing on the ground

        velocity = VELOCITIES.get(command)
        if velocity is not None:
            self.drone.send_rc_control(*velocity)

    def stop(self):
        """Zero every channel - the drone holds position."""
        self.drone.send_rc_control(0, 0, 0, 0)


if __name__ == "__main__":
    import time

    link = DroneLink()
    link.send("takeoff")
    time.sleep(3)

    # Each command is held for two seconds. The drone should move continuously
    # for the whole two seconds, not hop a fixed distance and stop.
    for command in ("forward", "back", "left", "right", "ascend", "descend"):
        print("testing:", command)
        link.send(command)
        time.sleep(2)
        link.send("hover")
        time.sleep(1)

    link.send("land")
    time.sleep(3)
    print("Test routine finished.")