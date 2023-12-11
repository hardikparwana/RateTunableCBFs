import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.Acc_simple_rtcbf import *
# from robot_models.DoubleIntegrator2D import *

# figure
plt.ion()
fig = plt.figure()
ax = plt.axes()
ax.set_xlabel('time (s)')
ax.set_title('Velocity')

fig2, ax2 = plt.subplots(3,1)
ax2[0].set_title('Barrier')
ax2[0].set_xlabel('time (s)')
ax2[1].set_title('Input')
ax2[1].set_xlabel('time (s)')
ax2[2].set_title('Velocity')
ax2[2].set_xlabel('time (s)')

# sim parameters
dt = 0.05
tf = 50#8
d_min = 0.3
T = int( tf/dt )
alpha_nom = 0.2#2000.0
k = 10#1000#2000.0

alpha11 = 0.5
alpha12 = 0.7

alpha21 = 0.5
alpha31 = 0.5

# Robot
# robot = ACC2D(np.array([20,13.89,100]), dt, ax, id = 0, color = 'g' ) #18,10,150
robot = ACC2D(np.array([20,10,40]), dt, ax, id = 0, color = 'g' ) #18,10,150

# Controller
max_u = robot. af * robot.m * robot.gr / 2
robot.g()

ca = cp.Parameter(value=robot.af)
cd = cp.Parameter(value=robot.af)
u2 = cp.Variable((1,1))
u2_ref = cp.Parameter((1,1),value = np.zeros((1,1)) )
Fr = cp.Parameter()
slack = cp.Variable((1,1))
slack_coeff = cp.Parameter((4,1), value=np.array([[1],[0],[0],[0]]))
num_constraints2  = 4
A2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
b2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
const2 = [A2 @ u2 + slack_coeff @ slack  >= b2]
# const2 += [ cp.abs( u2[0,0] ) <= max_u ]
const2 += [ u2[0,0]<=  ca * robot.gr ] #* robot.m
const2 += [ u2[0,0]>= -cd * robot.gr ] #* robot.m
# objective2 = cp.Minimize( 1.0/robot.m**2 * cp.sum_squares(u2) -2*Fr/robot.m**2 * u2 + robot.psc * cp.sum_squares(slack))
objective2 = cp.Minimize( 1.0 * cp.sum_squares(u2) -2*Fr * u2 + 1000 * cp.sum_squares(slack))
# objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref  ) + 10000 * cp.sum_squares(slack) )
cbf_controller2 = cp.Problem( objective2, const2 )

# Alpha lower bound
# max_u = robot. af * robot.m * robot.gr / 2
# u2_lb = cp.Variable((1,1))
# num_constraints2_lb  = 1
# alpha2_lb = cp.Variable((num_constraints2_lb,1))
# alpha2_ref_lb = cp.Parameter((num_constraints2_lb,1), value = alpha_nom*np.ones((num_constraints2_lb,1)))
# Fr_lb = cp.Parameter()
# A2_lb = cp.Parameter((num_constraints2_lb,1),value=np.zeros((num_constraints2_lb,1)))
# A2_alpha_lb = cp.Parameter((num_constraints2_lb,num_constraints2_lb),value=alpha_nom*np.eye(num_constraints2_lb))
# b2_lb = cp.Parameter((num_constraints2_lb,1),value=np.zeros((num_constraints2_lb,1)))
# const2_lb = [A2_lb @ u2_lb  + A2_alpha_lb @ alpha2_lb >= b2_lb]
# const2_lb += [ cp.abs( u2_lb[0,0] ) <= max_u ]
# objective2_lb = cp.Minimize( cp.sum_squares( alpha2_lb - alpha2_ref_lb ) )
# cbf_controller2_lb = cp.Problem( objective2_lb, const2_lb )

# Simulation
# ax.axhline(robot.X[1,0], label='Leader velocity')
ax.axhline(robot.vd, color='r', label='Ego Desired velocity')
ax.legend()
ax2[2].axhline(robot.vmax, color='r')
ax2[2].axhline(robot.vmin, color='r')
ax.set_xlim([-1,15])

aL = 0

hp = ax2[0].scatter([], [], c='b')
up = ax2[1].scatter([], [], c='orange')
cap = ax2[1].scatter([], [], c='k')
cdp = ax2[1].scatter([], [], c='k')
velp = ax2[2].scatter([], [],c='orange')

hs = []
us = []
cas = []
cds = []
vels = []
ts = []

for t in range(T):
    
    aL = 0#- 0.6 * np.tanh(200/(t+dt))

    V, dV_dx = robot.lyapunov()
    h1, h1_dot, dh1_dot_dx = robot.distance_barrier()
    h2, dh2_dx = robot.vel_barrier1()
    h3, dh3_dx = robot.vel_barrier2()
    
    u2_ref.value[0,0] = - 1 * (robot.X[0,0] - robot.vd)
    Fr.value = robot.Fr()
    
    # CLF
    A2.value[0,0] = - dV_dx @ robot.g()[:,0].reshape(-1,1)
    b2.value[0,0] = k * V + dV_dx @ robot.f()

    
    # CBF1 for controller
    A2.value[1,0] = dh1_dot_dx @ robot.g()[:,0].reshape(-1,1)
    b2.value[1,0] = -dh1_dot_dx @ robot.f() - (alpha11+alpha12)*h1_dot - alpha11*alpha12*h1 - dh1_dot_dx @ robot.g()[:,1].reshape(-1,1) * aL   # alpha2_lb.value * h
    
    # CBF1 for controller
    A2.value[2,0] = (dh2_dx @ robot.g()[:,0].reshape(-1,1))
    b2.value[2,0] = -dh2_dx @ robot.f() -alpha21*h2#- alpha2_lb.value * h
    
    # CBF1 for controller
    A2.value[3,0] = (dh3_dx @ robot.g()[:,0].reshape(-1,1))
    b2.value[3,0] = -dh3_dx @ robot.f() - alpha31*h3 #- alpha2_lb.value * h
    
    # cbf_controller2.solve( solver=cp.GUROBI, reoptimize=True )
    try:
        cbf_controller2.solve( solver=cp.GUROBI, reoptimize=True )
        if not (cbf_controller2.status=='optimal' or cbf_controller2.status=='optimal_inaccurate'):
            print(f"CBF-QP infeasible: {cbf_controller2.status}")
            break
            # exit()
    except Exception as e:
        # break
        print(f"error: {e}")
        cbf_controller2.solve( solver=cp.GUROBI, reoptimize=True, verbose=True )
        plt.ioff()
        plt.show()
        exit()
        
    # robot.step( u2.value )
    robot.step( np.array([ [u2.value[0,0]], [aL] ]) )
        
    # print(f"t: {t}, alpha: {alpha2_lb.value} v: {robot.X[0,0]}, u:{u2.value}, slack: {slack.value}, const: {A2.value[1,0]*u2.value[0,0]-b2.value[1,0]}")
    
    # ax.scatter(t*dt, robot.X[0,0], c='b')
    
    hs.append(h1)
    us.append(u2.value[0,0])
    cas.append(ca.value * robot.gr)
    cds.append(-cd.value * robot.gr)
    vels.append(robot.X[0,0])
    ts.append(t*dt)

    print(f"{u2.value}, {u2_ref.value}, h:{h1}, vel:{robot.X[0,0]}, leadervel:{robot.X[1,0]}")
    
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    # fig2.canvas.draw()
    # fig2.canvas.flush_events()

print(f"END")

plt.ioff()

ax2[0].plot(ts, hs, 'orange')

ax.plot(ts, robot.Xs[0,1:], 'k')
ax.plot(ts, robot.Xs[1,1:], 'b')

ax2[0].plot(ts, hs, c='b')
ax2[1].plot(ts, us, c='orange')
ax2[1].plot(ts, cas, c='k')
ax2[1].plot(ts, cds, c='k')
ax2[2].plot(ts, vels,c='orange')


plt.show()

