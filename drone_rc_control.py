from pysimverse import Drone
import time

drone = Drone()
drone.connect()
drone.take_off()

left_right= 20
forward_backward = 0
up_down = 0
yaw=0

start_time = time.time()
#for continuous motion
while time.time()- start_time<5:
    drone.send_rc_control(left_right, forward_backward, up_down, yaw)
    time.sleep(0.1)

drone.send_rc_control(0,0,0,0)

drone.land()
time.sleep(1)