import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator3D import *
from robot_models.SingleIntegrator6D import *
from robot_models.Unicycle2D import *


# figure
plt.ion()
fig = plt.figure()#(dpi=100)
# fig.set_size_inches(33, 15)
ax = plt.axes(projection ="3d",xlim=(-5,5),ylim=(-5,5), zlim=(-0.01,4.0))   
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1,1,4.0/10])

# Set ground plane
length = 5
height = -0.1
x = [-length,-length,length,length]
y = [-length,length,length,-length]
z = [-height,height,height,height]
verts = [list(zip(x,y,z))]
# ax.add_collection3d(Poly3DCollection(verts,facecolor='gray', alpha=0.5))

dt = 0.01
robots = []

cone_angle = np.pi/6       
height = 2.0

robots.append( SingleIntegrator6D(np.array([-1.5,1.5,height,0,0,0]), dt, ax, id = 0, cone_length = height/np.cos(cone_angle), cone_angle = cone_angle) )
robots.append( SingleIntegrator6D(np.array([1.5,-1.0,height,0,0,0]), dt, ax, id = 1, cone_length = height/np.cos(cone_angle), cone_angle = cone_angle) )

robots.append( SingleIntegrator3D(np.array([1,1,0]), dt, ax, id = 2, color = 'g' ) )
robots.append( SingleIntegrator3D(np.array([2,4,0]), dt, ax, id = 3, color = 'g' ) )
robots.append( SingleIntegrator3D(np.array([-2,-4,0]), dt, ax, id = 4, color = 'g' ) )

robots.append( Unicycle2D( np.array([-1,-1,0,0]), dt, ax, id = 5, color='b' ) )
robots.append( Unicycle2D( np.array([4,2,0,np.pi]), dt, ax, id = 6, color='b' ) )
robots.append( Unicycle2D( np.array([-4,-2,0,np.pi/3]), dt, ax, id = 7, color='b' ) )

num_robots = len(robots)

### Controllers


### 1. Single Integrator 2D & Unicycle 2D
u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints1  = num_robots - 1
A1 = cp.Parameter((num_constraints1,2),value=np.zeros((num_constraints1,2)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
slack_constraints1 = cp.Parameter( (num_constraints1,1), value = np.zeros((num_constraints1,1)) )
const1 = [A1 @ u1 >= b1 + slack_constraints1]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref  ) + 100 * cp.sum_squares( slack_constraints1 ) )
cbf_controller1 = cp.Problem( objective1, const1 )

### 2. Single Integrator 3D

u2 = cp.Variable((3,1))
u2_ref = cp.Parameter((3,1),value = np.zeros((3,1)) )
num_constraints2  = num_robots - 1
A2 = cp.Parameter((num_constraints2,3),value=np.zeros((num_constraints2,3)))
b2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
slack_constraints2 = cp.Parameter( (num_constraints2,2), value = np.zeros((num_constraints2,2)) )
const2 = [A2 @ u2 >= b2 + slack_constraints2]
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref ) + 100 * cp.sum_squares( slack_constraints2 ) )
cbf_controller2 = cp.Problem( objective2, const2 )

###

# plt.show()
for i in range(100):
    
    
    
    
    robots[0].step(np.array([0.0,0.0,0.0,0.0,0,0]))
    robots[2].step(np.array([1.0,0.0,0.0]))
    robots[5].step(robots[5].nominal_input( np.array([1,1]) ))
    
    for i in range(num_robots):
        
        # Get constraints: Ax >= b
        robots[i].A1 = robots[i]
    
    
    fig.canvas.draw()
    fig.canvas.flush_events()