import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.DoubleIntegrator3D import *
from robot_models.Surveillance import *
from robot_models.Unicycle2D import *
from robot_models.obstacles import *

# figure
plt.ion()
fig = plt.figure()#(dpi=100)
# fig.set_size_inches(33, 15)
ax = plt.axes(projection ="3d",xlim=(-2,2),ylim=(-2,2), zlim=(-0.01,4.0))   
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1,1,4.0/10])

alpha = 2.0
tf = 20.0
dt = 0.1
# robot = Unicycle2D( np.array([0,0,0,0]), dt, ax, id = 5, color='b', alpha = alpha, mode='ego', target = np.array([3,3]).reshape(-1,1))
# obs = circle(1.5, 1.5, 0, 1.0, ax, 0)

robot = DoubleIntegrator3D( np.array([-1,-1.1,0,0.1,0,0]), dt, ax, id = 5, color='b', alpha = alpha, mode='ego', target = np.array([2,-1, 0]).reshape(-1,1))
obs = circle(0.5, -1, 0, 0.3, ax, 0)

u3 = cp.Variable((3,1))
u3_ref = cp.Parameter((3,1),value = np.zeros((3,1)) )
A3 = cp.Parameter((1,3),value=np.zeros((1,3)))
b3 = cp.Parameter((1,1),value=np.zeros((1,1)))
slack_constraints3 = cp.Parameter( (1,1), value = np.zeros((1,1)) )
const3 = [A3 @ u3 >= b3 + slack_constraints3]
objective3 = cp.Minimize( cp.sum_squares( u3 - u3_ref  ) )
cbf_controller3 = cp.Problem( objective3, const3 )

# plt.show()

for t in range(int(tf/dt)):
    
    h, dh_dxi, dh_dxj = robot.agent_barrier( obs, obs.radius )
    
    A3.value[0,:] = dh_dxi @ robot.g() 
    b3.value[0,:] = - dh_dxi @ robot.f() - alpha * h
    u3_ref.value = robot.nominal_input( robot.target, 'none' )
    
    cbf_controller3.solve(solver=cp.GUROBI)
    # print(f"u3ref:{u3_ref.value.T}, u3:{u3.value}")
    robot.step( u3.value )
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    
    
    
    