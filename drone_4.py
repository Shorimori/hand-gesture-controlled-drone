from pysimverse import Drone
import time

drone = Drone()
drone.connect()

drone.take_off()
drone.set_speed(200)
drone.move_left(350)
drone.move_forward(400)
drone.land()
drone.take_off()

drone.move_backward(100)
drone.move_down(100)
drone.move_right(800)




drone.land()
