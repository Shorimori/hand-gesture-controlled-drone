from pysimverse import Drone
import time
#default speed: 20 cm/s
# connecting the drone
drone = Drone()
drone.connect()

drone.take_off()       # go up 1m = 100cm
time.sleep(3)


drone.set_speed(50)
drone.move_down(20)        #20 cm
time.sleep(1)
drone.move_up(30)
time.sleep(1)


drone.move_forward(40)
time.sleep(5)
drone.move_right(40)
time.sleep(3)

drone.move_forward(20)
time.sleep(3)
drone.move_backward(40)
time.sleep(3)


drone.rotate(90)
time.sleep(5)


drone.land()
time.sleep(3)



