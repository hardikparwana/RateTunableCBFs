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
dt = 0.01
tf = 5.5
d_min = 0.3
T = int( tf/dt )
alpha1_nom = 1.0#0.5#1.0
alpha2_nom = 2.0#1.0

# Obstacles/Uncooperative agents
obs = []
obs.append( SingleIntegrator2D(np.array([-2,0]), dt, ax, id = 0, color = 'k' ) )
obs.append( SingleIntegrator2D(np.array([ 3,0]), dt, ax, id = 1, color = 'k' ) )
alpha1 = alpha1_nom * np.ones(len(obs))
alpha1_dot = 0 * alpha1 
alpha1_bound = 0 * alpha1

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
# const2 += [ alpha2 == alpha2_ref ]
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref  ) + 1000*cp.sum_squares(alpha2-alpha2_ref) )
cbf_controller2 = cp.Problem( objective2, const2 )

# Simulation
update = True
# update = False
print(f"t: {-1}, alpha1: {alpha1}, alpha2: {alpha2}")


def simulate_scenario(robot_x):
    
    obs = []
    obs.append( SingleIntegrator2D(np.array([-2,0]), dt, ax, id = 0, color = 'k' ) )
    obs.append( SingleIntegrator2D(np.array([ 3,0]), dt, ax, id = 1, color = 'k' ) )
    robot = DoubleIntegrator2D(np.array([robot_x,0.0,0,0]), dt, ax, id = 0, color = 'g' )
    
    alpha1 = alpha1_nom * np.ones(len(obs))
    alpha1_dot = 0 * alpha1 
    alpha1_bound = 0 * alpha1
    alpha2.value = alpha2_nom * np.ones((num_constraints2,1))
    alpha2_ref.value = alpha2_nom*np.ones((num_constraints2,1))

    for t in range(T):
        # print(f"x:{robot.X.T}")
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
            try:
                assert(h>=0.01)
            except Exception as e:
                print(f"h assert error")
                return t
            try:
                assert(h_dot + alpha1[j]*h>=0.01)
            except Exception as e:
                print(f"psi assert error")
                return t
            A2.value[j,:] = dh_dot_dxi @ robot.g()
            A2_alpha.value[j,:] = h_dot + alpha1[j] * h
            b2.value[j,:] = -dh_dot_dxi @ robot.f() - dh_dot_dxj @ obs[j].xdot - alpha1[j] * h_dot #- alpha1_dot[j] * h
        v_ref = -1.5 * ( robot.X[0:2] - goal )
        u2_ref.value = - 1.5 * ( robot.X[2:4] - v_ref )
        # print(f"alpha1: {alpha1}, alpha1_bound: {alpha1_bound}")
        try:
            cbf_controller2.solve( solver=cp.GUROBI, reoptimize=True )
            if cbf_controller2.status!='optimal' and cbf_controller2.status!='optimal_inaccurate':
                print(f"CBF-QP infeasible: {cbf_controller2.status}")
                return t
        except Exception as e:
            print(f"error: {e}")
            cbf_controller2.solve( solver=cp.GUROBI, reoptimize=True, verbose=True )
            return t
            
        robot.step( u2.value )  
        
        
        # print(f"t: {t}, alpha1: {alpha1}, alpha2: {alpha2.value[:,0]}, x: {robot.X.T}")
            
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # ax2[0].scatter(t*dt, u2.value[0,0], c='g')
        # ax2[1].scatter(t*dt, u2.value[1,0], c='r')
        
        # fig2.canvas.draw()
        # fig2.canvas.flush_events()
    return T
    
    
simulate_scenario(-1.0)

# update = True
# xs = np.linspace(-1.95,2.95,60)
# tsim = []
# for x in xs:
#     tsim.append(simulate_scenario(x)*dt)
    
# fig2, ax2 = plt.subplots()
# ax2.plot(xs,tsim, 'g*', label='Algorithm 1')
# # plt.show()

# print(f"\n default case now \n ")
# update = False

# const2 += [ alpha2 == alpha2_ref ]
# cbf_controller2 = cp.Problem( objective2, const2 )

# tsim2 = []
# for x in xs:
#     tsim2.append(simulate_scenario(x)*dt)
    
# plt.ioff()
# # ax3, fig3 = plt.subplots()
# ax2.plot(xs,tsim2, 'r*', label='Fixed Parameters')
# ax2.legend()
# ax2.set_xlabel('time (s)')
# ax2.set_ylabel('Time of first infeasible QP')

# plt.show()    


