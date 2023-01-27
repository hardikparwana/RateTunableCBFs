import numpy as np

def wrap_angle(angle):
    if angle>np.pi:
        return  angle - 2*np.pi
    elif angle<-np.pi:
        return  angle + 2*np.pi 
    else:
        return angle
    
def quat_to_rot(quat):
    return np.array([  [ (quat[0]**2+quat[1]**2-quat[2]**2-quat[3]**2), 2*(quat[1]*quat[2]-quat[0]*quat[3]), 2*(quat[0]*quat[2]+quat[1]*quat[3]) ],
                       [ 2*(quat[1]*quat[2]+quat[0]*quat[3]), (quat[0]**2-quat[1]**2+quat[2]**2-quat[3]**2), 2*(quat[2]*quat[3]-quat[0]*quat[1])  ],
                       [ 2*(quat[1]*quat[3]-quat[0]*quat[2]), 2*(quat[0]*quat[1]+quat[2]*quat[3]), (quat[0]**2-quat[1]**2-quat[2]**2+quat[3]**2) ]       
    ])
    
def quat_multiple(q1, q2):
    return np.array([
                [q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q[3]*q[3]],
                [q1[1]*q2[0] + q1[0]*q2[1] + q1[2]*q2[3] - q[3]*q[2]],
                [q1[2]*q2[0] + q1[0]*q2[2] + q1[3]*q2[1] - q[1]*q[3]],
                [q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q[2]*q[1]]
    ])