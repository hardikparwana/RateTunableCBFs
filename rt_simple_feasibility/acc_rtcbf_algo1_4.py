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

name = "test1_4_"

fig2, ax2 = plt.subplots(7,1)
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

alpha11 = 0.3 #0.5
alpha11_dot = 0
alpha11_nom = 0.3 #0.5

alpha12 = 0.7
alpha12_nom = 0.7

alpha21 = 0.5
alpha21_nom = 0.5

alpha31 = 0.5
alpha31_nom = 0.5

# update = False
update = True

# Robot
# robot = ACC2D(np.array([20,13.89,100]), dt, ax, id = 0, color = 'g' ) #18,10,150
robot = ACC2D(np.array([20,10,100]), dt, ax, id = 0, color = 'g' ) #18,10,150

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
A2_alpha = cp.Parameter((num_constraints2,num_constraints2), value=alpha11*np.zeros((num_constraints2,num_constraints2)))
alpha2_cp = cp.Variable((4,1), value=np.zeros((4,1)))
alpha2_ref_cp = cp.Parameter((4,1), value=alpha11*np.zeros((4,1)))
b2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
const2 = [A2 @ u2 + A2_alpha @ alpha2_cp + slack_coeff @ slack  >= b2]
const2 += [ u2[0,0]<=  ca * robot.gr / 2 ] #* robot.m
const2 += [ u2[0,0]>= -cd * robot.gr / 2] #* robot.m
const2 += [ alpha2_cp >= 0 ]
if not update:
    const2 += [alpha2_cp == alpha2_ref_cp]
    
# objective2 = cp.Minimize( 1.0 * cp.sum_squares(u2) -2*Fr * u2 / 100 + 100 * cp.sum_squares(slack)/2000 + 10000 * cp.sum_squares(alpha2_cp[0:2] - alpha2_ref_cp[0:2]) + 2000000 * cp.sum_squares(alpha2_cp[2:4] - alpha2_ref_cp[2:4])  )
objective2 = cp.Minimize( 1.0 * cp.sum_squares(u2) -2*Fr * u2 / 100 + 100 * cp.sum_squares(slack)/2000 + 20000 * cp.sum_squares(alpha2_cp[0:2] - alpha2_ref_cp[0:2]) + 100 * cp.sum_squares(alpha2_cp[2:4] - alpha2_ref_cp[2:4])  )


cbf_controller2 = cp.Problem( objective2, const2 )

alpha2_ref_cp.value[1,0] = alpha12
alpha2_ref_cp.value[2,0] = alpha21
alpha2_ref_cp.value[3,0] = alpha31

alpha2_cp.value[1,0] = alpha12
alpha2_cp.value[2,0] = alpha21
alpha2_cp.value[3,0] = alpha31

# Simulation
# ax.axhline(robot.X[1,0], label='Leader velocity')
ax.axhline(robot.vd, color='r', label='Ego Desired velocity')
ax.legend()
ax2[2].axhline(robot.vmax, color='r')
ax2[2].axhline(robot.vmin, color='r')
# ax.set_xlim([-1,15])

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
alpha11s = []
alpha12s = []
alpha21s = []
alpha31s = []

alpha2_ref_cp.value[1,0] = alpha12
alpha2_ref_cp.value[2,0] = alpha21
alpha2_ref_cp.value[3,0] = alpha31

alpha11_bound = 0

for t in range(T):
    
    if t*dt<15: ##10:
    # aL = -0.6
        aL = - 0.6 * (1 - np.tanh(  0.5/(10.1-t*dt) ) )
        # print(f"{np.tanh(  0.0/(20.1-t*dt) )}, aL: {aL}")
    else:
        aL = 0.0
        
        
    alpha2_ref_cp.value[1,0] = alpha12_nom
    alpha2_ref_cp.value[2,0] = alpha21_nom
    alpha2_ref_cp.value[3,0] = alpha31_nom
    
    # aL = - 0.6 * np.tanh(200/(t+dt))

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
    A2_alpha.value[1,1] = h1_dot + alpha11 * h1
    b2.value[1,0] = -dh1_dot_dx @ robot.f() - alpha11 * h1_dot - alpha11_dot * h1 - dh1_dot_dx @ robot.g()[:,1].reshape(-1,1) * aL   # alpha2_lb.value * h
    
    # CBF1 for controller
    A2.value[2,0] = (dh2_dx @ robot.g()[:,0].reshape(-1,1))
    A2_alpha.value[2,2] = h2
    b2.value[2,0] = -dh2_dx @ robot.f() #-alpha21*h2#- alpha2_lb.value * h
    
    # CBF1 for controller
    A2.value[3,0] = (dh3_dx @ robot.g()[:,0].reshape(-1,1))
    A2_alpha.value[3,3] = h3
    b2.value[3,0] = -dh3_dx @ robot.f() #- alpha31*h3 #- alpha2_lb.value * h
    
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
    print(f"alpha11: {alpha11}, alpha12: {alpha2_cp.value[1,0]}, {alpha11_bound}")
    
    robot.step( np.array([ [u2.value[0,0]], [aL] ]) )
    
    
    if update:
        h1, h1_dot, dh1_dot_dx = robot.distance_barrier()
        alpha11_bound = - h1_dot / h1
        
        if alpha11_nom >= alpha11_bound:
            alpha11 = alpha11_nom
        else:
            alpha11 = alpha11_bound
        # if alpha11 >= alpha11_bound:
        #     alpha11_des = alpha11
        #     offset = 1.0
        #     if (alpha11>alpha11_nom+offset) and (alpha11-offset>alpha11_bound):
        #         alpha11_des = alpha11-offset
        #     elif (alpha11<alpha11_nom-offset) and (alpha11+offset>alpha11_bound):
        #         alpha11_des = alpha11+offset
        # else:
        #     alpha11_des = 1.1 * alpha11_bound
            
        # alpha11_change = alpha11_des - alpha11    
        # alpha11_ddot = (alpha11_change - alpha11_dot*dt)*2/dt**2
        # alpha11_dot = alpha11_dot + alpha11_ddot * dt
        # alpha11 = alpha11_des
        
    # print(f"t: {t}, alpha: {alpha2_lb.value} v: {robot.X[0,0]}, u:{u2.value}, slack: {slack.value}, const: {A2.value[1,0]*u2.value[0,0]-b2.value[1,0]}")
    
    # ax.scatter(t*dt, robot.X[0,0], c='b')
    
    hs.append(h1)
    us.append(u2.value[0,0])
    cas.append(ca.value * robot.gr)
    cds.append(-cd.value * robot.gr)
    vels.append(robot.X[0,0])
    ts.append(t*dt)
    alpha11s.append(alpha11)
    alpha12s.append(alpha2_cp.value[1,0])
    alpha21s.append(alpha2_cp.value[2,0])
    alpha31s.append(alpha2_cp.value[3,0])

    # print(f"{u2.value}, {u2_ref.value}, h:{h1}, vel:{robot.X[0,0]}, leadervel:{robot.X[1,0]}")
    
    # print(f"t: {t*dt}, aL: {aL}, vL {robot.X[1,0]} alpha11: {alpha11}, alpha12: {alpha2_cp.value[1,0]}, alpha21: {alpha2_cp.value[2,0]}, alpha31: {alpha2_cp.value[3,0]}, alpha21ref: {alpha2_ref_cp.value[2,0]}")
    
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

ax2[3].plot(ts, alpha11s)
ax2[4].plot(ts, alpha12s)
ax2[3].plot(ts, alpha21s)
ax2[3].plot(ts, alpha31s)

fig3, ax3 = plt.subplots()
ax3.plot(ts, robot.Xs[0,1:], 'k', label='Agent Velocity')
ax3.plot(ts, robot.Xs[1,1:], 'b', label='Leader velocity')
ax3.axhline(robot.vmax, color='k', linestyle='--')
ax3.axhline(robot.vmin, color='k', linestyle='--')
ax3.legend()

fig3.savefig(name+'acc_vel.png')
fig3.savefig(name+'acc_vel.eps')

# control input
fig7, ax7 = plt.subplots()
ax7.plot(ts, us, 'k', label='Control Input')
ax7.axhline(ca.value * robot.gr, color='k', linestyle='--')
ax7.axhline(-cd.value * robot.gr, color='k', linestyle='--')
ax7.set_xlabel('time (s)')
ax7.legend()

fig7.savefig(name+'acc_input.png')
fig7.savefig(name+'acc_input.eps')

fig4, ax4 = plt.subplots()
ax4.plot(ts, hs, 'k', label='barrier function')
ax4.legend()

fig4.savefig(name+'acc_barrier.png')
fig4.savefig(name+'acc_barrier.eps')

fig5, ax5 = plt.subplots()
ax5.plot(ts, alpha11s, 'k', label='alpha11')
ax5.plot(ts, alpha21s, 'r', label='alpha21')
ax5.plot(ts, alpha31s, 'g', label='alpha31')
ax5.legend()

fig5.savefig(name+'acc_alpha1.png')
fig5.savefig(name+'acc_alpha1.eps')

fig6, ax6 = plt.subplots()
ax6.plot(ts, alpha12s, 'k', label='alpha12')
ax6.legend()

fig6.savefig(name+'acc_alpha2.png')
fig6.savefig(name+'acc_alpha2.eps')






plt.show()

