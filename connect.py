"""
connect.py - ArduPilot / MAVLink

Replaces the PySimverse version.

Works against ArduPilot SITL now.
"""

import math
import time

from pymavlink import mavutil



CONNECTION = "udp:127.0.0.1:14550"

TAKEOFF_ALT = 5.0     # metres
SPEED = 1.0           # m/s for horizontal and vertical movement
MAX_YAW_RATE = 0.6    # rad/s when yaw input is at full scale


VELOCITIES = {
    "hover":   (0.0, 0.0, 0.0),
    "forward": (SPEED, 0.0, 0.0),
    "back":    (-SPEED, 0.0, 0.0),
    "left":    (0.0, -SPEED, 0.0),
    "right":   (0.0, SPEED, 0.0),
    "ascend":  (0.0, 0.0, -SPEED),
    "descend": (0.0, 0.0, SPEED),
}


IGNORE_POS = 0b0000000000000111
IGNORE_ACC = 0b0000000111000000
IGNORE_YAW = 0b0000010000000000
TYPE_MASK = IGNORE_POS | IGNORE_ACC | IGNORE_YAW


class DroneLink:
  

    def __init__(self, connection=CONNECTION):
        print(f"connecting to {connection} ...")
        self.master = mavutil.mavlink_connection(connection)
        self.master.wait_heartbeat()
        print(f"heartbeat from system {self.master.target_system}")

        self._armed = False
        self._alt = 0.0
        self._takeoff_sent = False

        # Ask for the streams we need to know what the vehicle is doing.
        self.master.mav.request_data_stream_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)

 

    def poll(self):
        """
        Drain pending telemetry. Non-blocking, so it is safe to call from the
        command loop. Keeps _armed and _alt current.
        """
        while True:
            msg = self.master.recv_match(
                type=["HEARTBEAT", "GLOBAL_POSITION_INT"], blocking=False)
            if msg is None:
                return
            if msg.get_type() == "HEARTBEAT":
                self._armed = bool(msg.base_mode &
                                   mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            else:
                self._alt = msg.relative_alt / 1000.0   # mm -> m

    @property
    def airborne(self):
        """Armed and actually off the ground."""
        self.poll()
        return self._armed and self._alt > 0.5

  
    def set_mode(self, mode):
        self.master.set_mode(self.master.mode_mapping()[mode])

    def arm(self):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1, 0, 0, 0, 0, 0, 0)

    def takeoff(self, altitude=TAKEOFF_ALT):
        """
        GUIDED mode, arm, climb. Returns once the vehicle is airborne or the
        attempt times out - it blocks, which is exactly why the app runs this
        on the worker thread.
        """
        self.set_mode("GUIDED")
        time.sleep(0.5)

        self.arm()
        deadline = time.time() + 10
        while time.time() < deadline:
            self.poll()
            if self._armed:
                break
            time.sleep(0.2)
        else:
            print("[drone] arming failed - check prearm messages in SITL")
            return

        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, altitude)

        deadline = time.time() + 30
        while time.time() < deadline:
            self.poll()
            if self._alt >= altitude * 0.9:
                print(f"[drone] airborne at {self._alt:.1f} m")
                return
            time.sleep(0.3)
        print(f"[drone] takeoff timed out at {self._alt:.1f} m")

    def land(self):
        self.stop()
        self.set_mode("LAND")
  
    def send_velocity(self, vx, vy, vz, yaw_rate=0.0):
        """One velocity setpoint. Returns immediately."""
        self.master.mav.set_position_target_local_ned_send(
            0,                                        # time_boot_ms
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            TYPE_MASK,
            0, 0, 0,           # position - ignored
            vx, vy, vz,        # velocity m/s
            0, 0, 0,           # acceleration - ignored
            0, yaw_rate)       # yaw angle ignored, yaw rate used

    def stop(self):
        self.send_velocity(0.0, 0.0, 0.0, 0.0)

  
    def send(self, command, yaw=0):
      
        if command == "takeoff":
            if not self.airborne and not self._takeoff_sent:
                self._takeoff_sent = True
                self.takeoff()
            return

        if command == "land":
            if self.airborne:
                self.land()
            self._takeoff_sent = False
            return

        if not self.airborne:
            return

        velocity = VELOCITIES.get(command)
        if velocity is None:
            return

        vx, vy, vz = velocity
        yaw_rate = (yaw / 100.0) * MAX_YAW_RATE
        self.send_velocity(vx, vy, vz, yaw_rate)


if __name__ == "__main__":
    link = DroneLink()

    print("\ntaking off ...")
    link.send("takeoff")

    def hold(command, seconds, yaw=0):
        print(f"  {command:8s} for {seconds}s  (yaw={yaw})")
        end = time.time() + seconds
        while time.time() < end:
            link.send(command, yaw)
            time.sleep(0.05)

    for command in ("forward", "back", "left", "right", "ascend", "descend"):
        hold(command, 3)
        hold("hover", 1)

    print("\nturning on the spot ...")
    hold("hover", 3, yaw=60)

    print("\nlanding ...")
    link.send("land")
    time.sleep(10)
    print("done.")
