import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator3D import *
from robot_models.Surveillance import *
from robot_models.Unicycle2D import *
from robot_models.obstacles import *

# figure
plt.ion()
fig = plt.figure()#(dpi=100)
# fig.set_size_inches(33, 15)
ax = plt.axes(projection ="3d",xlim=(-1,5),ylim=(-1,5), zlim=(-0.01,4.0))   
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1,1,4.0/10])

alpha = 0.8
tf = 20.0
dt = 0.1
# robot = Unicycle2D( np.array([0,0,0,0]), dt, ax, id = 5, color='b', alpha = alpha, mode='ego', target = np.array([3,3]).reshape(-1,1))
# obs = circle(1.5, 1.5, 0, 1.0, ax, 0)

robot = Unicycle2D( np.array([-1,-1.1,0,0]), dt, ax, id = 5, color='b', alpha = alpha, mode='ego', target = np.array([3,-1]).reshape(-1,1))
obs = circle(1.5, -1, 0, 0.3, ax, 0)

u2 = cp.Variable((2,1))
u2_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
A2 = cp.Parameter((1,2),value=np.zeros((1,2)))
b2 = cp.Parameter((1,1),value=np.zeros((1,1)))
slack_constraints2 = cp.Parameter( (1,1), value = np.zeros((1,1)) )
const2 = [A2 @ u2 >= b2 + slack_constraints2]
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref  ) )
cbf_controller2 = cp.Problem( objective2, const2 )

# plt.show()

for t in range(int(tf/dt)):
    
    h, dh_dxi, dh_dxj = robot.agent_barrier( obs, obs.radius )
    
    A2.value[0,:] = dh_dxi @ robot.g() 
    b2.value[0,:] = - dh_dxi @ robot.f() - alpha * h
    u2_ref.value = robot.nominal_input( robot.target, 'none' )
    cbf_controller2.solve(solver=cp.GUROBI)
    
    robot.step( u2.value )
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    
    
    
    