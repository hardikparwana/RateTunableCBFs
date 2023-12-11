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
tf = 5.5
d_min = 0.1#0.3
T = int( tf/dt )
alpha_nom = 1.0

# Robot
goal = np.array([6,0]).reshape(-1,1)

obs = []
obs.append( SingleIntegrator2D(np.array([-2,0]), dt, ax, id = 0, color = 'k' ) )
obs.append( SingleIntegrator2D(np.array([ 3,0]), dt, ax, id = 1, color = 'k' ) )

# Controller
max_u = 1.5
u2 = cp.Variable((2,1))
u2_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints2  = len(obs)
alpha2 = cp.Variable((num_constraints2,1), value=alpha_nom*np.ones((num_constraints2,1)))
alpha2_ref = cp.Parameter((num_constraints2,1), value=alpha_nom*np.ones((num_constraints2,1)))
A2 = cp.Parameter((num_constraints2,2),value=np.zeros((num_constraints2,2)))
A2_alpha = cp.Parameter((num_constraints2,num_constraints2),value=alpha_nom*np.eye(num_constraints2))
b2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
const2 = [A2 @ u2 + A2_alpha @ alpha2 >= b2]
const2 += [ cp.abs( u2[0,0] ) <= max_u ]
const2 += [ cp.abs( u2[1,0] ) <= max_u ]
const2 += [ alpha2 >= 0 ]
# const2 += [ alpha2 == alpha2_ref ]
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref  ) + 10000*cp.sum_squares(alpha2 - alpha2_ref) )
cbf_controller2 = cp.Problem( objective2, const2 )

# Simulation


def simulate_scenario(alpha_nom, alpha2_nom):
    
    # Obstacles/Uncooperative agents
    obs = []
    obs.append( SingleIntegrator2D(np.array([-2,0]), dt, ax, id = 0, color = 'k' ) )
    obs.append( SingleIntegrator2D(np.array([ 3,0]), dt, ax, id = 1, color = 'k' ) )
    robot = SingleIntegrator2D(np.array([-0.5,0]), dt, ax, id = 0, color = 'g' )
    alpha = alpha_nom * np.ones(len(obs))
    alpha_bound = [0] * len(obs)
    alpha2.value = alpha_nom * np.ones((num_constraints2,1))
    alpha2.value[0,0] = alpha_nom#  alpha_nom * np.ones((num_constraints2,1))
    alpha2.value[1,0] = alpha2_nom#
    alpha2_ref.value[0,0] = alpha_nom#
    alpha2_ref.value[1,0] = alpha2_nom#
    # alpha2.value = alpha_nom * np.ones((num_constraints2,1))
    # alpha2_ref.value = alpha_nom*np.ones((num_constraints2,1))
    
    for t in range(T):
        
        if t*dt<=3.5:
            obs[0].step( np.array([1.5, 0.0]) )  #step( np.array([0.5+1.0*np.exp(-t*dt*0.), 0.0]) ) # left
            obs[1].step( np.array([0.5, 0.0]) ) # right 
        elif t*dt<4.0:
            obs[0].step( np.array([1.5 - 1.0*( np.tanh( (t*dt-3.75)/(3.5-t*dt)/(t*dt-4.0) )/2+0.5 ), 0.0]) )
            obs[1].step( np.array([0.5, 0.0]) ) # right 
        else: 
            obs[0].step( np.array([0.5, 0.0]) )  #
            obs[1].step( np.array([0.5, 0.0]) ) # right 
        
        for j in range(len(obs)):
            
            if alpha2.value[j,0] > alpha_nom + 0.1:
                alpha2_ref.value[j,0] = alpha2.value[j,0] - 0.1
            elif alpha2.value[j,0] < alpha_nom - 0.1:
                alpha2_ref.value[j,0] = alpha2.value[j,0] + 0.1
            # else:
            #     alpha2_ref.value[j,0] = alpha2.value[j,0]
            
            h, dh_dxi, dh_dxj = robot.obstacle_barrier(obs[j], d_min)
            A2.value[j,:] = dh_dxi @ robot.g()
            b2.value[j,:] = -dh_dxi @ robot.f() - dh_dxj @ obs[j].xdot 
            A2_alpha.value[j,j] = h ##alpha[j] * h
        u2_ref.value = - 1.5 * ( robot.X - goal )
        # print(f"A2:{A2.value}, b2:{b2.value}, Aalpha:{A2_alpha.value}")
        # exit()
        try:
            cbf_controller2.solve( solver=cp.GUROBI, reoptimize=True )
            if cbf_controller2.status!='optimal':
                print(f"CBF-QP infeasible: {cbf_controller2.status}")
                return t, robot.Xs
                # exit()
        except Exception as e:
            print(f"error: {e}")
            cbf_controller2.solve( solver=cp.GUROBI, reoptimize=True, verbose=True )
            return t, robot.Xs
            # exit()
            
        robot.step( u2.value )
        
        print(f"t: {t}, alpha: {alpha2.value.T}")
        
    return T, robot.Xs
        
        # fig.canvas.draw()
        # fig.canvas.flush_events()


# simulate_scenario(0)
fig2, ax2 = plt.subplots()
alpha1s = [0.5, 1.0, 1.5, 2.0, 0.5, 1.5]
alpha2s = [0.5, 1.0, 1.5, 2.0, 2.0, 0.5]
cc = ['r', 'g', 'c', 'y', 'm', 'b']
# tsim = []

obs = []
obs.append( SingleIntegrator2D(np.array([-2,0]), dt, ax, id = 0, color = 'k' ) )
obs.append( SingleIntegrator2D(np.array([ 3,0]), dt, ax, id = 1, color = 'k' ) )
for t in range(T):    
    if t*dt<=3.5:
        obs[0].step( np.array([1.5, 0.0]) )  #step( np.array([0.5+1.0*np.exp(-t*dt*0.), 0.0]) ) # left
        obs[1].step( np.array([0.5, 0.0]) ) # right 
    elif t*dt<4.0:
        obs[0].step( np.array([1.5 - 1.0*( np.tanh( (t*dt-3.75)/(3.5-t*dt)/(t*dt-4.0) )/2+0.5 ), 0.0]) )
        obs[1].step( np.array([0.5, 0.0]) ) # right 
    else: 
        obs[0].step( np.array([0.5, 0.0]) )  #
        obs[1].step( np.array([0.5, 0.0]) ) # right 
ax2.plot(np.linspace(0,T*dt,T), obs[0].Xs[0,1:], color='k', linewidth=4)
ax2.plot(np.linspace(0,T*dt,T), obs[1].Xs[0,1:], color='k', linewidth=4)
            
            

# ax2.plot()
for j in range(len(alpha1s)):
    ts, Xs = simulate_scenario(alpha1s[j], alpha2s[j])
    ax2.plot(np.linspace(0,ts*dt,ts), Xs[0,1:], color=cc[j], alpha=0.5, label=r'$\nu_1^1=$'+f"{alpha1s[j]}, " + r'$\nu_i^2$='+f"{alpha2s[j]}")
    
print(f" default test ")
    
update = False
const2 += [ alpha2 == alpha2_ref ]
cbf_controller2 = cp.Problem( objective2, const2 )

for j in range(len(alpha1s)):
    ts, Xs = simulate_scenario(alpha1s[j], alpha2s[j])
    ax2.plot(np.linspace(0,ts*dt,ts), Xs[0,1:], color=cc[j], linestyle='--')
    
ax2.set_xlabel('time (s)')
ax2.set_ylabel('X')
ax2.legend()    

fig2.savefig("si_stats_diff_param_traj.png")
fig2.savefig("si_stats_diff_param_traj.eps")
    
plt.ioff()
plt.show()


# plt.show()
    

