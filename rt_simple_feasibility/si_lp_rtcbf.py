import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator2D_rtcbf import *
# from robot_models.DoubleIntegrator2D import *

# figure
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-5,10),ylim=(-5,5))   
ax.set_xlabel('X')
ax.set_ylabel('Y')

# sim parameters
dt = 0.05
tf = 8
d_min = 0.3
T = int( tf/dt )
alpha_nom = 0.5#1.0

# Obstacles/Uncooperative agents
obs = []
obs.append( SingleIntegrator2D(np.array([-2,0]), dt, ax, id = 0, color = 'k' ) )
obs.append( SingleIntegrator2D(np.array([ 3,0]), dt, ax, id = 1, color = 'k' ) )
# obs.append( SingleIntegrator2D(np.array([ 0,3]), dt, ax, id = 2, color = 'k' ) )
# obs.append( SingleIntegrator2D(np.array([ 0,-3]), dt, ax, id = 3, color = 'k' ) )
alpha = alpha_nom * np.ones(len(obs))
alpha_bound = [0] * len(obs)

alpha_der_max = 0.01
min_dist = 0.4
h_min = 0.5

# Robot
robot = SingleIntegrator2D(np.array([0,0]), dt, ax, id = 0, color = 'g' )
goal = np.array([6,0]).reshape(-1,1)

# Controller
max_u = 500 
u2 = cp.Variable((2,1))
u2_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints2  = len(obs)
alpha2 = cp.Variable((num_constraints2,1))
alpha2_ref = cp.Parameter((num_constraints2,1), value=alpha_nom*np.ones((num_constraints2,1)))
A2 = cp.Parameter((num_constraints2,2),value=np.zeros((num_constraints2,2)))
A2_alpha = cp.Parameter((num_constraints2,num_constraints2),value=alpha_nom*np.eye(num_constraints2))
b2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
const2 = [A2 @ u2 + A2_alpha @ alpha2 >= b2]
const2 += [ cp.abs( u2[0,0] ) <= max_u ]
const2 += [ cp.abs( u2[1,0] ) <= max_u ]
const2 += [ alpha2 >= 0 ]
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref  ) + cp.sum_squares(alpha2 - alpha2_ref) )
cbf_controller2 = cp.Problem( objective2, const2 )

# Subsystem Controller
Q_sub = cp.Parameter((1,2), value=np.zeros((1,2)))
u2_sub = cp.Variable((2,1))
num_constraints2_sub  = len(obs)
delta_sub = cp.Variable((num_constraints2_sub,1))
A2_sub = cp.Parameter((num_constraints2_sub,2),value=np.zeros((num_constraints2_sub,2)))
b2_sub = cp.Parameter((num_constraints2_sub,1),value=np.zeros((num_constraints2_sub,1)))
const2_sub = [A2_sub @ u2_sub + delta_sub >= b2_sub]
# const2_sub = [A2_sub @ u2_sub >= b2_sub]
const2_sub += [ cp.abs( u2_sub[0,0] ) <= max_u ]
const2_sub += [ cp.abs( u2_sub[1,0] ) <= max_u ]
objective2_sub = cp.Minimize( - Q_sub @ u2_sub + 1000 * cp.sum_squares(delta_sub) )
# objective2_sub = cp.Minimize( - Q_sub @ u2_sub )
cbf_controller2_sub = cp.Problem( objective2_sub, const2_sub )

# Simulation

for t in range(T):
    
    obs[0].step( np.array([1.5, 0.0]) )  #step( np.array([0.5+1.0*np.exp(-t*dt*0.), 0.0]) ) # left
    obs[1].step( np.array([0.5, 0.0]) ) # right 
    # obs[2].step( np.array([0.9, -0.9]) ) # top
    # obs[3].step( np.array([0.9, 0.4]) ) # bottom
    # obs[2].step( np.array([0.9, -0.9]) ) # top
    # obs[3].step( np.array([0.9, 0.0]) ) # bottom
    
    for j in range(len(obs)):
        h, dh_dxi, dh_dxj = robot.obstacle_barrier(obs[j], d_min)
        A2.value[j,:] = dh_dxi @ robot.g()
        b2.value[j,:] = -dh_dxi @ robot.f() - dh_dxj @ obs[j].xdot 
        A2_alpha.value[j,j] = h ##alpha[j] * h
    u2_ref.value = - 1.5 * ( robot.X - goal )
            
    # for j in range(len(obs)):
    #     h, dh_dxi, dh_dxj = robot.obstacle_barrier(obs[j], d_min)
    #     A2.value[j,:] = dh_dxi @ robot.g()
    #     b2.value[j,:] = -dh_dxi @ robot.f() - alpha[j] * h - dh_dxj @ obs[j].xdot
    # u2_ref.value = - 1.5 * ( robot.X - goal )
    
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
    
    print(f"t: {t}, alpha: {alpha2.value.T}")
        
    fig.canvas.draw()
    fig.canvas.flush_events()
    


