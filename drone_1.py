from pysimverse import Drone
import time

drone = Drone()
drone.connect()
drone.take_off()

left_right= 20
forward_backward = 0
up_down = 0
yaw=0

#for continuous motion
while True:
    drone.send_rc_control(left_right, forward_backward, up_down, yaw)

drone.land()
drone.sleep(1)