from pysimverse import Drone
import time
import cv2

drone = Drone()
drone.connect()
drone.streamon() #to turn on the aerial camera

drone.take_off()

seconds=3
while seconds>0:
    frame, is_success = drone.get_frame()
    cv2.imshow(winname='frame',mat = frame)
    cv2.waitKey(0)
    time.sleep(3)
    seconds-=1
    
    
# press any key 3 times 
drone.land()
time.sleep(3)

