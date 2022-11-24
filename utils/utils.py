import numpy as np

def wrap_angle(angle):
    if angle>np.pi:
        return  angle - 2*np.pi
    elif angle<-np.pi:
        return  angle + 2*np.pi 
    else:
        return angle