import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.Acc_rtcbf import *
# from robot_models.DoubleIntegrator2D import *

# figure
plt.ion()
fig = plt.figure()
ax = plt.axes()
ax.set_xlabel('time (s)')
ax.set_title('Velocity')

fig2, ax2 = plt.subplots(1,1)
ax2.set_title('Barrier')
ax2.set_xlabel('time (s)')

# sim parameters
dt = 0.05
tf = 8
d_min = 0.3
T = int( tf/dt )
alpha = 0.2#2000.0
k = 1000#2000.0

# Robot
robot = ACC2D(np.array([22,5,150]), dt, ax, id = 0, color = 'g' ) #18,10,150

# Controller
max_u = robot. af * robot.m * robot.gr / 2
u2 = cp.Variable((1,1))
u2_ref = cp.Parameter((1,1),value = np.zeros((1,1)) )
Fr = cp.Parameter()
slack = cp.Variable((1,1))
slack_coeff = cp.Parameter((2,1), value=np.array([[1],[0]]))
num_constraints2  = 2
A2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
b2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
const2 = [A2 @ u2 + slack_coeff @ slack >= b2]
const2 += [ cp.abs( u2[0,0] ) <= max_u ]
# objective2 = cp.Minimize( 1.0/robot.m**2 * cp.sum_squares(u2) -2*Fr/robot.m**2 * u2 + robot.psc * cp.sum_squares(slack))
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref  ) + 10000 * cp.sum_squares(slack) )
cbf_controller2 = cp.Problem( objective2, const2 )

# Simulation
ax.axhline(robot.X[1,0], label='Leader velocity')
ax.axhline(robot.vd, color='r', label='Ego Desired velocity')
ax.legend()
for t in range(T):

    V, dV_dx = robot.lyapunov()
    h, dh_dx = robot.distance_barrier()
    u2_ref = - 1 * (robot.X[0,0] - robot.vd)
    Fr.value = robot.Fr()
    
    # CLF
    A2.value[0,0] = - dV_dx @ robot.g()
    b2.value[0,0] = k * V + dV_dx @ robot.f()
    
    # CBF
    A2.value[1,0] = (dh_dx @ robot.g())
    b2.value[1,0] = (-dh_dx @ robot.f() - alpha * h)
    
    try:
        cbf_controller2.solve( solver=cp.GUROBI, reoptimize=True )
        if cbf_controller2.status!='optimal':
            print(f"CBF-QP infeasible: {cbf_controller2.status}")
            exit()
    except Exception as e:
        print(f"error: {e}")
        cbf_controller2.solve( solver=cp.GUROBI, reoptimize=True, verbose=True )
        exit()
        
    robot.step( u2.value )
        
    print(f"t: {t}, v: {robot.X[0,0]}, u:{u2.value}, slack: {slack.value}, const: {A2.value[1,0]*u2.value[0,0]-b2.value[1,0]}")
    
    ax.scatter(t*dt, robot.X[0,0], c='b')
    ax2.scatter(t*dt, h, c='b')
        
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig2.canvas.draw()
    fig2.canvas.flush_events()


