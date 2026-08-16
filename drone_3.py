from pysimverse import Drone
import time

drone= Drone()
drone.connect()

drone.take_off()
drone.set_speed(200)

drone.move_left(230)
drone.move_forward(90)
drone.land()
drone.take_off()

drone.move_right(230)
drone.move_forward(90)

drone.land()
drone.take_off()
drone.move_right(280)
drone.move_forward(140)

drone.land()
