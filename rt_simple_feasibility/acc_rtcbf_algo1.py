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

fig2, ax2 = plt.subplots(2,1)
ax2[0].set_title('Barrier')
ax2[0].set_xlabel('time (s)')
ax2[1].set_title('Input')
ax2[1].set_xlabel('time (s)')

# sim parameters
dt = 0.05
tf = 8
d_min = 0.3
T = int( tf/dt )
alpha_nom = 0.2#2000.0
k = 1000#2000.0

# Robot
robot = ACC2D(np.array([22,5,100]), dt, ax, id = 0, color = 'g' ) #18,10,150

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
const2 = [A2 @ u2 + slack_coeff @ slack  >= b2]
const2 += [ cp.abs( u2[0,0] ) <= max_u ]
# objective2 = cp.Minimize( 1.0/robot.m**2 * cp.sum_squares(u2) -2*Fr/robot.m**2 * u2 + robot.psc * cp.sum_squares(slack))
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref  ) + 10000 * cp.sum_squares(slack) )
cbf_controller2 = cp.Problem( objective2, const2 )

# Alpha lower bound
max_u = robot. af * robot.m * robot.gr / 2
u2_lb = cp.Variable((1,1))
num_constraints2_lb  = 1
alpha2_lb = cp.Variable((num_constraints2_lb,1))
alpha2_ref_lb = cp.Parameter((num_constraints2_lb,1), value = alpha_nom*np.ones((num_constraints2_lb,1)))
Fr_lb = cp.Parameter()
A2_lb = cp.Parameter((num_constraints2_lb,1),value=np.zeros((num_constraints2_lb,1)))
A2_alpha_lb = cp.Parameter((num_constraints2_lb,num_constraints2_lb),value=alpha_nom*np.eye(num_constraints2_lb))
b2_lb = cp.Parameter((num_constraints2_lb,1),value=np.zeros((num_constraints2_lb,1)))
const2_lb = [A2_lb @ u2_lb  + A2_alpha_lb @ alpha2_lb >= b2_lb]
const2_lb += [ cp.abs( u2_lb[0,0] ) <= max_u ]
objective2_lb = cp.Minimize( cp.sum_squares( alpha2_lb - alpha2_ref_lb ) )
cbf_controller2_lb = cp.Problem( objective2_lb, const2_lb )

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
    
    # CBF for lb optimization
    A2_lb.value[0,0] = (dh_dx @ robot.g())
    A2_alpha_lb.value[0,0] = h
    b2_lb.value[0,0] = -dh_dx @ robot.f()
    
    
    
    try:
        cbf_controller2_lb.solve( solver=cp.GUROBI, reoptimize=True )
        if cbf_controller2_lb.status!='optimal':
            print(f"CBF-QP infeasible: {cbf_controller2.status}")
            exit()
    except Exception as e:
        print(f"error: {e}")
        cbf_controller2_lb.solve( solver=cp.GUROBI, reoptimize=True, verbose=True )
        exit()
    
    # if alpha2_lb.value > alpha_nom + 0.1:
    #     alpha2_ref_lb.value = alpha2_lb.value - 0.1
    # elif alpha2_lb.value < alpha_nom - 0.1:
    #     alpha2_ref_lb.value = alpha2_lb.value + 0.1
    # else:
    #     alpha2_ref_lb.value = alpha2_lb.value
    
    # CBF for controller
    A2.value[1,0] = (dh_dx @ robot.g())
    b2.value[1,0] = -dh_dx @ robot.f() - alpha2_lb.value * h
    
    try:
        cbf_controller2.solve( solver=cp.GUROBI, reoptimize=True )
        if cbf_controller2.status!='optimal':
            print(f"CBF-QP infeasible: {cbf_controller2.status}")
            exit()
    except Exception as e:
        print(f"error: {e}")
        cbf_controller2.solve( solver=cp.GUROBI, reoptimize=True, verbose=True )
        plt.ioff()
        plt.show()
        exit()
        
    robot.step( u2.value )
        
    print(f"t: {t}, alpha: {alpha2_lb.value} v: {robot.X[0,0]}, u:{u2.value}, slack: {slack.value}, const: {A2.value[1,0]*u2.value[0,0]-b2.value[1,0]}")
    
    ax.scatter(t*dt, robot.X[0,0], c='b')
    ax2[0].scatter(t*dt, h, c='b')
    ax2[1].scatter(t*dt,u2.value, c='orange')
        
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig2.canvas.draw()
    fig2.canvas.flush_events()



plt.ioff()
plt.show()

