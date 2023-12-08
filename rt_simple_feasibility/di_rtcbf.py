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

# sim parameters
dt = 0.05
tf = 8
d_min = 0.3
T = int( tf/dt )
alpha1_nom = 0.5#1.0
alpha2_nom = 2.0#1.0

# Obstacles/Uncooperative agents
obs = []
obs.append( SingleIntegrator2D(np.array([-2,0]), dt, ax, id = 0, color = 'k' ) )
obs.append( SingleIntegrator2D(np.array([ 3,0]), dt, ax, id = 1, color = 'k' ) )
obs.append( SingleIntegrator2D(np.array([ 0,3]), dt, ax, id = 2, color = 'k' ) )
obs.append( SingleIntegrator2D(np.array([ 0,-3]), dt, ax, id = 3, color = 'k' ) )
alpha1 = alpha1_nom * np.ones(len(obs))
alpha2 = alpha2_nom * np.ones(len(obs))
alpha1_bound = [0] * len(obs)
alpha2_bound = [0] * len(obs)

alpha_der_max = 0.01
min_dist = 0.4
h_min = 0.5

# Robot
robot = DoubleIntegrator2D(np.array([0,0.0,0,0]), dt, ax, id = 0, color = 'g' )
goal = np.array([6,0]).reshape(-1,1)

# Controller
max_u = 3#500 
u2 = cp.Variable((2,1))
u2_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints2  = len(obs)
A2 = cp.Parameter((num_constraints2,2),value=np.zeros((num_constraints2,2)))
b2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
const2 = [A2 @ u2 >= b2]
const2 += [ cp.abs( u2[0,0] ) <= max_u ]
const2 += [ cp.abs( u2[1,0] ) <= max_u ]
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref  ) )
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
update = True
print(f"t: {-1}, alpha1: {alpha1}, alpha2: {alpha2}")

for t in range(T):
    
    obs[0].step( np.array([1.5, 0.0]) )  #step( np.array([0.5+1.0*np.exp(-t*dt*0.), 0.0]) ) # left
    obs[1].step( np.array([0.5, 0.0]) ) # right 
    # obs[2].step( np.array([0.9, -0.9]) ) # top
    # obs[3].step( np.array([0.9, 0.4]) ) # bottom
    # obs[2].step( np.array([0.9, -0.9]) ) # top
    # obs[3].step( np.array([0.9, 0.0]) ) # bottom
    
    for j in range(len(obs)):
        h, h_dot, dh_dot_dxi, dh_dot_dxj = robot.obstacle_barrier(obs[j], d_min)
        
        # alpha1 update
        if update:
            alpha1_bound[j] = - h_dot / h
            if alpha1[j]<=alpha1_bound[j]:
                alpha1[j] = 1.9*alpha1_bound[j]        
        
        A2.value[j,:] = dh_dot_dxi @ robot.g()
        b2.value[j,:] = -dh_dot_dxi @ robot.f() - dh_dot_dxj @ obs[j].xdot - (alpha1[j] + alpha2[j]) * h_dot - alpha1[j] * alpha2[j] * h
    v_ref = -1.5 * ( robot.X[0:2] - goal )
    u2_ref.value = - 1.5 * ( robot.X[2:4] - v_ref )
    
    # subsystems update new
    if update:
        for j in range(len(obs)):
            A2_sub.value = np.copy(A2.value)
            b2_sub.value = np.copy(b2.value)
            A2_sub.value[j,:] = 0 * A2_sub.value[j,:]
            b2_sub.value[j,:] = 0 * b2_sub.value[j,:]
            h, h_dot, dh_dot_dxi, dh_dot_dxj = robot.obstacle_barrier(obs[j], d_min)
            Q_sub.value = dh_dot_dxi @ robot.g()
            cbf_controller2_sub.solve(solver=cp.GUROBI, reoptimize=True)
            if cbf_controller2_sub.status!='optimal':
                print(f"Suboproblem CBF-LP infeasible: {cbf_controller2_sub.status}")
                exit()
            h2_dot = dh_dot_dxi @ ( robot.f() + robot.g() @ u2_sub.value) + dh_dot_dxj @ obs[j].xdot + alpha1[j] * h_dot
            h2 = h_dot + alpha1[j] * h
            alpha2_bound[j] = - h2_dot / h2
            if alpha2[j] < alpha2_bound[j]:
                alpha2[j] = alpha2_bound[j] 
            
   
    
            
    for j in range(len(obs)):
        h, h_dot, dh_dot_dxi, dh_dot_dxj = robot.obstacle_barrier(obs[j], d_min)
        A2.value[j,:] = dh_dot_dxi @ robot.g()
        b2.value[j,:] = -dh_dot_dxi @ robot.f() - dh_dot_dxj @ obs[j].xdot - (alpha1[j] + alpha2[j]) * h_dot - alpha1[j] * alpha2[j] * h
    
    try:
        cbf_controller2.solve( solver=cp.GUROBI, reoptimize=True )
        if cbf_controller2.status!='optimal' and cbf_controller2.status!='optimal_inaccurate':
            print(f"CBF-QP infeasible: {cbf_controller2.status}")
            exit()
    except Exception as e:
        print(f"error: {e}")
        cbf_controller2.solve( solver=cp.GUROBI, reoptimize=True, verbose=True )
        exit()
        
    robot.step( u2.value )
    
    
    
    print(f"t: {t}, alpha1: {alpha1}, alpha2: {alpha2}")
        
    fig.canvas.draw()
    fig.canvas.flush_events()
    


