import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator2D_rtcbf import *
from robot_models.DoubleIntegrator2D_rtcbf import *
# from robot_models.DoubleIntegrator2D import *

# figure
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-5,10),ylim=(-5,5))   
ax.set_xlabel('X')
ax.set_ylabel('Y')

fig2, ax2 = plt.subplots(2,1)
ax2[0].set_ylabel('Control Input 1')
ax2[1].set_ylabel('Control Input 2')

# sim parameters
dt = 0.05
tf = 8
d_min = 0.3
T = int( tf/dt )
alpha1_nom = 1.0#0.5#1.0
alpha2_nom = 2.0#1.0

# Obstacles/Uncooperative agents
obs = []
obs.append( SingleIntegrator2D(np.array([-2,0]), dt, ax, id = 0, color = 'k' ) )
obs.append( SingleIntegrator2D(np.array([ 3,0]), dt, ax, id = 1, color = 'k' ) )
# obs.append( SingleIntegrator2D(np.array([ 0,3]), dt, ax, id = 2, color = 'k' ) )
# obs.append( SingleIntegrator2D(np.array([ 0,-3]), dt, ax, id = 3, color = 'k' ) )
alpha1 = alpha1_nom * np.ones(len(obs))
# alpha2 = alpha2_nom * np.ones(len(obs))
alpha1_dot = 0 * alpha1 
alpha1_dot_prev = 0 * alpha1
alpha1_bound = 0 * alpha1

alpha2_dot = 0 * alpha2

# Robot
robot = DoubleIntegrator2D(np.array([0,0.0,0,0]), dt, ax, id = 0, color = 'g' )
goal = np.array([6,0]).reshape(-1,1)

# Controller
max_u = 3#500 
u2 = cp.Variable((2,1))
u2_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints2  = len(obs)
alpha2 = cp.Variable((num_constraints2,1), value=alpha2_nom*np.ones((num_constraints2,1)))
alpha2_ref = cp.Parameter((num_constraints2,1), value=alpha2_nom*np.ones((num_constraints2,1)))
A2 = cp.Parameter((num_constraints2,2),value=np.zeros((num_constraints2,2)))
A2_alpha = cp.Parameter((num_constraints2,num_constraints2),value=alpha2_nom*np.ones((num_constraints2,num_constraints2)))
b2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
const2 = [A2 @ u2 + A2_alpha @ alpha2 >= b2]
const2 += [ cp.abs( u2[0,0] ) <= max_u ]
const2 += [ cp.abs( u2[1,0] ) <= max_u ]
const2 += [ alpha2 == alpha2_ref ]
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref  ) + 1000*cp.sum_squares(alpha2-alpha2_ref) )
cbf_controller2 = cp.Problem( objective2, const2 )

# Simulation
update = False#True
print(f"t: {-1}, alpha1: {alpha1}, alpha2: {alpha2}")

for t in range(T):
    
    obs[0].step( np.array([1.5, 0.0]) )  #step( np.array([0.5+1.0*np.exp(-t*dt*0.), 0.0]) ) # left
    obs[1].step( np.array([0.5, 0.0]) ) # right 
    # obs[2].step( np.array([0.9, -0.9]) ) # top
    # obs[3].step( np.array([0.9, 0.4]) ) # bottom
    # obs[2].step( np.array([0.9, -0.9]) ) # top
    # obs[3].step( np.array([0.9, 0.0]) ) # bottom
    
    for j in range(len(obs)):
        
        
        if alpha2.value[j,0] > alpha2_nom + 0.1:
            alpha2_ref.value[j,0] = alpha2.value[j,0] - 0.1
        elif alpha2.value[j,0] < alpha2_nom - 0.1:
            alpha2_ref.value[j,0] = alpha2.value[j,0] + 0.1
        else:
            alpha2_ref.value[j,0] = alpha2.value[j,0]
        
        h, h_dot, dh_dot_dxi, dh_dot_dxj = robot.obstacle_barrier(obs[j], d_min)
        
        # alpha1 update
        if update:
            alpha1_bound[j] = - h_dot / h
            alpha_temp = alpha1[j]
            if alpha1[j]<=alpha1_bound[j]:
                alpha1[j] = 1.9*alpha1_bound[j]    
            # if h_dot < 0:
            #     alpha1[j] = 1.9*alpha1_bound[j]    
            # else:
            #     alpha1[j] = 0
            alpha1_dot[j] = (alpha1[j] - alpha_temp)/dt
        # print(f"j: {j}, h: {h}")
        A2.value[j,:] = dh_dot_dxi @ robot.g()
        A2_alpha.value[j,:] = h_dot + alpha1[j] * h
        b2.value[j,:] = -dh_dot_dxi @ robot.f() - dh_dot_dxj @ obs[j].xdot - alpha1[j] * h_dot - alpha1_dot[j] * h
    v_ref = -1.5 * ( robot.X[0:2] - goal )
    u2_ref.value = - 1.5 * ( robot.X[2:4] - v_ref )
    
    try:
        cbf_controller2.solve( solver=cp.GUROBI, reoptimize=True )
        if cbf_controller2.status!='optimal' and cbf_controller2.status!='optimal_inaccurate':
            print(f"CBF-QP infeasible: {cbf_controller2.status}")
            plt.ioff()
            plt.show()
            exit()
    except Exception as e:
        print(f"error: {e}")
        cbf_controller2.solve( solver=cp.GUROBI, reoptimize=True, verbose=True )
        plt.ioff()
        plt.show()
        exit()
        
    robot.step( u2.value )  
    
    
    print(f"t: {t}, alpha1: {alpha1}, alpha2: {alpha2.value[:,0]}")
        
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    ax2[0].scatter(t*dt, u2.value[0,0], c='g')
    ax2[1].scatter(t*dt, u2.value[1,0], c='r')
    
    fig2.canvas.draw()
    fig2.canvas.flush_events()
    
plt.ioff()
plt.show()
    

    


