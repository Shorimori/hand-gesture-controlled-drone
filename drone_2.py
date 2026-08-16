from pysimverse import Drone
import time

drone = Drone()
drone.connect()

drone.take_off()


drone.set_speed(100)
drone.move_forward(400)
drone.move_right(350)
drone.move_backward(20)






drone.land()
time.sleep(3)





