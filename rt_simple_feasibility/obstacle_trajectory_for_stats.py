import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator2D_rtcbf import *
from robot_models.DoubleIntegrator2D_rtcbf import *

from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 16})
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-5,6.5),ylim=(-6.5,6.5))   
ax.set_aspect(1)
ax.set_xlabel('X')
ax.set_ylabel('Y')

# sim parameters
dt = 0.02 #0.05
tf = 6.5 #8 #10 #5.5
d_min = 0.3 #0.5#0.3
T = int( tf/dt )

si = False
di = True
live = False

alpha_si = 1.5
alpha1_si = 1.5
alpha2_si = 3.0

robot_x = np.array([4,3.0,0,0]) # new3 control

robot = DoubleIntegrator2D(np.array([6,1.0,0,0]), dt, ax, id = 0, color = 'r' )
robot_goal = np.array([ -4,0 ]).reshape(-1,1)
ax.scatter( robot_goal[0], robot_goal[1], facecolor='none', edgecolor='r' )

obs = []
if si:
    obs.append( SingleIntegrator2D(np.array([-2,0]), dt, ax, id = 0, color = 'k' ) )
    obs.append( SingleIntegrator2D(np.array([ 3,6]), dt, ax, id = 1, color = 'k' ) )
    obs.append( SingleIntegrator2D(np.array([-2,4]), dt, ax, id = 0, color = 'k' ) )
    obs.append( SingleIntegrator2D(np.array([ 3,-4]), dt, ax, id = 1, color = 'k' ) )
    obs.append( SingleIntegrator2D(np.array([-2,-3]), dt, ax, id = 0, color = 'k' ) )
    obs.append( SingleIntegrator2D(np.array([ -3,-5]), dt, ax, id = 1, color = 'k' ) )
elif di:
    obs.append( DoubleIntegrator2D(np.array([-2,0,0,0]), dt, ax, id = 0, color = 'k' ) )
    obs.append( DoubleIntegrator2D(np.array([ 3,6,0,0]), dt, ax, id = 1, color = 'k' ) )
    obs.append( DoubleIntegrator2D(np.array([-2,4,0,0]), dt, ax, id = 0, color = 'k' ) )
    obs.append( DoubleIntegrator2D(np.array([ 3,-4,0,0]), dt, ax, id = 1, color = 'k' ) )
    obs.append( DoubleIntegrator2D(np.array([-2,-3,0,0]), dt, ax, id = 0, color = 'k' ) )
    obs.append( DoubleIntegrator2D(np.array([ -3,-5,0,0]), dt, ax, id = 1, color = 'k' ) )

goals = []
goals.append( np.array([2,0]).reshape(-1,1) )
goals.append( np.array([-3,-6]).reshape(-1,1) )
goals.append( np.array([2,-4]).reshape(-1,1) )
goals.append( np.array([-3,4]).reshape(-1,1) )
goals.append( np.array([-2,-3]).reshape(-1,1) )
goals.append( np.array([3,5]).reshape(-1,1) )
for i in range(len(goals)):
    ax.scatter(obs[i].X[0,0], obs[i].X[1,0], edgecolor='none', facecolor='k', s=200)
    ax.scatter(goals[i][0], goals[i][1], edgecolor='k', facecolor='none', s=300) #, marker='x')

# plt.ioff()
# plt.show()

# Obstacle CBF controllers
u2_si = cp.Variable((2,1))
u2_si_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints_si  = len(obs)-1
slack_si = cp.Variable((num_constraints_si,1))
A2_si= cp.Parameter((num_constraints_si,2),value=np.zeros((num_constraints_si,2)))
b2_si = cp.Parameter((num_constraints_si,2),value=np.zeros((num_constraints_si,2)))
const2_si = [A2_si @ u2_si + slack_si  >= b2_si]
const2_si += [ cp.abs(u2_si[0,0]) <= 3.0]
const2_si += [ cp.abs(u2_si[1,0]) <= 3.0]
objective2_si = cp.Minimize(  cp.sum_squares(u2_si-u2_si_ref) + 100 * cp.sum_squares(slack_si) )
problem_si = cp.Problem( objective2_si, const2_si )

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=12, metadata=metadata)

# plt.ioff()
# plt.show()

# exit()

for t in range(T):

    if si:
        for i in range(len(obs)):
            index = 0
            u2_si_ref.value = -1.5 * ( obs[i].X - goals[i] )
            # obs[i].step( u2_si_ref )
            for j in range(len(obs)):
                if i==j:
                    continue
                h, dh_dxi, dh_dxj = obs[i].obstacle_barrier(obs[j], d_min, prev_state=True)
                A2_si.value[index,:] = dh_dxi
                b2_si.value[index] = -alpha_si * h - dh_dxj @ obs[j].xdot #obs[j].U_prev
                index = index + 1
            problem_si.solve(solver=cp.GUROBI)
            obs[i].step( u2_si.value )
            if live:
                obs[i].render_plot()
    elif di:
        kx = 1.5 #0.8
        kv = 3.0 #2.0

        # other agent controller
        for i in range(len(obs)):
            index = 0
            vd = -kx * ( obs[i].X[0:2] - goals[i] )
            u2_si_ref.value = - kv * ( obs[i].X[2:4] - vd )  #-1.5 * ( obs[i].X - goals[i] )
            # obs[i].step( u2_si_ref )
            for j in range(len(obs)):
                if i==j:
                    continue
                h, h_dot, dh_dot_dxi, dh_dot_dxj = obs[i].obstacle_barrier(obs[j], d_min, prev_state=True, agent_type='DI')
                A2_si.value[index,:] = dh_dot_dxi @ obs[i].g()
                b2_si.value[index] =  - dh_dot_dxi @ obs[i].f() - dh_dot_dxj @ obs[j].xdot - (alpha1_si + alpha2_si)*h_dot - alpha1_si * alpha2_si * h
                index = index + 1
            problem_si.solve(solver=cp.GUROBI)
            obs[i].step( u2_si.value )
            if live:
                obs[i].render_plot()

        
        if live:
            fig.canvas.draw()
            fig.canvas.flush_events()

tp = np.linspace(0, tf, obs[0].Xs.shape[1])
tp = -1 + 2*np.asarray(tp)/tp[-1]
tp = tp/np.max(tp)
div = 10
cc = np.tan(np.asarray(tp[1::div]))

plt.ioff()
for i in range(len(obs)):
    # ax.plot(obs[i].Xs[0,:], obs[i].Xs[1,:], 'r')
    im = ax.scatter(obs[i].Xs[0,1::div], obs[i].Xs[1,1::div], c=cc, rasterized=True, cmap = 'winter')
fig.colorbar(im, ax=ax, label='Normalized time')

fig.savefig("obstacle_paths.png")
fig.savefig("obstacle_paths.eps")
fig.savefig("obstacle_paths.svg")

plt.show()
