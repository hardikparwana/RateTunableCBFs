import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator2D_rtcbf import *
from robot_models.DoubleIntegrator2D_rtcbf import *

from matplotlib.animation import FFMpegWriter

plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-10,7),ylim=(-7,7))   
ax.set_xlabel('X')
ax.set_ylabel('Y')
# name = "new_di_di_stats_standard_proposed_alpha1_dot10.0_a1factor1.0_"
name = "new_di_di_stats_proposed_alpha1_dot10.0_alpha2dot1.0_a1factor1.5_"
# name = "new_di_di_stats_fixed_"
# name = "test_new_di_di_stats_fixed_"
# name = "stats+standard_proposed_alpha1_"

update = True
proposed = True
standard = False
live = False

max_alpha1_dot = 10.0 #0.5 #1
max_alpha2_dot = 1.0 #2.0 #2.0 #0.2 #1.0 #0.5 #1
alpha11_factor = 1.5 #1.5 #1.5

# t=2.6

# sim parameters
dt = 0.05
tf = 10 #5.5
d_min = 0.3 #0.5#0.3
T = int( tf/dt )
alpha1_nom = 4 #1.8#0.5#1.0
alpha2_nom = 8 #10 #5#1.0
alpha1_si = 1.5
alpha2_si = 3.0

robot = DoubleIntegrator2D(np.array([6,1.0,0,0]), dt, ax, id = 0, color = 'r' )
# robot = SingleIntegrator2D(np.array([6,1.0]), dt, ax, id = 0, color = 'r' )
robot_goal = np.array([ -4,0 ]).reshape(-1,1)
ax.scatter( robot_goal[0], robot_goal[1], facecolor='none', edgecolor='r' )


obs = []
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
    ax.scatter(goals[i][0], goals[i][1], edgecolor='g', facecolor='none')

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


u2 = cp.Variable((2,1))
u2_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints  = len(obs)
alpha2 = cp.Variable((num_constraints, 1), value=np.zeros((num_constraints, 1)))
alpha2_ref = cp.Parameter((num_constraints, 1), value=alpha2_nom*np.ones((num_constraints, 1)))
# slack = cp.Variable((num_constraints,1))
A2= cp.Parameter((num_constraints,2),value=np.zeros((num_constraints,2)))
A2_alpha = cp.Parameter((num_constraints,num_constraints),value=np.zeros((num_constraints,num_constraints)))
b2 = cp.Parameter((num_constraints,2),value=np.zeros((num_constraints,2)))
const2 = [A2 @ u2 + A2_alpha @ alpha2 >= b2]
const2 += [ cp.abs(u2[0,0]) <= 3.0]
const2 += [ cp.abs(u2[1,0]) <= 3.0]
const2 += [ alpha2 >= 0]
if not ( (proposed==True) or (update==True) or (standard==True) ) :
    const2 += [ alpha2 == alpha2_ref]
objective2 = cp.Minimize(  cp.sum_squares(u2-u2_ref) + 10000 * cp.sum_squares(alpha2-alpha2_ref) ) #+ 100 * cp.sum_squares(slack) )
problem2 = cp.Problem( objective2, const2 )
# alpha = 1.5

alpha1 = alpha1_nom * np.ones(len(obs))
alpha1_dot = 0.0 * np.ones(len(obs))

alpha2s = np.zeros((len(obs),1))
# update = True

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=12, metadata=metadata)

def simulate(robot_x):

    alpha2.value = np.zeros((num_constraints, 1))
    # alpha2.value = alpha2_si * np.ones((num_constraints, 1))

    robot = DoubleIntegrator2D(robot_x, dt, ax, id = 0, color = 'r' )
    # robot = SingleIntegrator2D(np.array([6,1.0]), dt, ax, id = 0, color = 'r' )
    robot_goal = np.array([ -4,0 ]).reshape(-1,1)
    ax.scatter( robot_goal[0], robot_goal[1], facecolor='none', edgecolor='r' )


    obs = []
    obs.append( DoubleIntegrator2D(np.array([-2,0, 0, 0]), dt, ax, id = 0, color = 'k' ) )
    obs.append( DoubleIntegrator2D(np.array([ 3,6, 0, 0]), dt, ax, id = 1, color = 'k' ) )
    obs.append( DoubleIntegrator2D(np.array([-2,4, 0, 0]), dt, ax, id = 0, color = 'k' ) )
    obs.append( DoubleIntegrator2D(np.array([ 3,-4, 0, 0]), dt, ax, id = 1, color = 'k' ) )
    obs.append( DoubleIntegrator2D(np.array([-2,-3, 0, 0]), dt, ax, id = 0, color = 'k' ) )
    obs.append( DoubleIntegrator2D(np.array([ -3,-5, 0, 0]), dt, ax, id = 1, color = 'k' ) )

    alpha1 = alpha1_nom * np.ones(len(obs))
    alpha1_dot = 0.0 * np.ones(len(obs))

    alpha2s = np.zeros((len(obs),1))
    update = True #True

    for t in range(T):

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


        kx = 1.0 #1.5 #0.8
        kv = 3.0 #2.0
        # Ego agent controller: METHOD 1
        vd = -kx * ( robot.X[0:2] - robot_goal )
        u2_ref.value = -kv * ( robot.X[2:4] - vd )
        h_min = 100
        psi_min = 100
        for j in range(len(obs)):

            h, h_dot, dh_dot_dxi, dh_dot_dxj = robot.obstacle_barrier(obs[j], d_min, prev_state=True, agent_type='DI') 
            if h<0:
                print(f"h={h}")
            if update:
                # max_alpha1_dot = 1.0 #0.5 #1
                # max_alpha2_dot = 1.0 #2.0 #2.0 #0.2 #1.0 #0.5 #1
                #porposed 1: 1.0
                # proposed2: 0.5

                alpha1_old = alpha1[j]
                alpha1_bound = - h_dot / max(h, 0.001)
                # alpha11_factor = 1.0 #1.5 #1.5

                if alpha1_nom >= alpha11_factor * alpha1_bound:
                    alpha1[j] = alpha1_nom
                else:
                    alpha1[j] = alpha11_factor * alpha1_bound
                alpha1_dot[j] = (alpha1[j]-alpha1_old)/dt
                alpha1_dot[j] = np.clip( alpha1_dot[j], -max_alpha1_dot, max_alpha1_dot )
                alpha1[j] = alpha1_old + alpha1_dot[j] * dt

                if proposed==True:
                    alpha2_offset = max_alpha2_dot  * dt
                    if alpha2.value[j,0] > alpha2_nom + alpha2_offset:
                        alpha2_ref.value[j,0] = alpha2.value[j,0] - alpha2_offset
                    elif alpha2.value[j,0] < alpha2_nom - alpha2_offset:
                        alpha2_ref.value[j,0] = alpha2.value[j,0] + alpha2_offset


            h_min = min(h_min, h)
            A2.value[j,:] = dh_dot_dxi @ robot.g()
            A2_alpha.value[j,j] = max(h_dot + alpha1[j] * h, 0.0)
            psi_min = min( psi_min, A2_alpha.value[j,j] )
            b2.value[j,:] = -dh_dot_dxi @ robot.f() - dh_dot_dxj @ obs[j].xdot - alpha1[j] * h_dot - alpha1_dot[j] * h
        # print(f"B: t: {t*dt}, alpha2: {alpha2.value.T}, h_min: {h_min}, psi_min: {psi_min}")    
        # print(f"alpha2: {alpha2.value}")
        problem2.solve(solver=cp.GUROBI) #, reoptimize=True)    
        if not (problem2.status=='optimal' or problem2.status=='optimal_inaccurate'):
            print(f"ERROR at t= {t*dt}")
            return t*dt
            break
        alpha2s = np.append( alpha2s, alpha2.value, axis=1 )
        # print(f"A: t: {t*dt}, alpha2: {alpha2.value.T}, h_min: {h_min}, psi_min: {psi_min}")
        robot.step( u2.value )
        if live:
            robot.render_plot()

        # Ego agent controller: single integrator
        # vd = -kx * ( robot.X[0:2] - robot_goal )
        # u2_ref.value = vd
        # for j in range(len(obs)):
        #     h, dh_dxi, dh_dxj = robot.obstacle_barrier(obs[j], d_min) 
        #     A2.value[j,j] = dh_dxi @ robot.g()
        #     A2_alpha.value[j,:] = h
        #     b2.value[j,:] = -dh_dxi @ robot.f() - dh_dxj @ obs[j].xdot
        # problem2.solve(solver=cp.GUROBI) #, reoptimize=True)
        # if not (problem2.status=='optimal' or problem2.status=='optimal_inaccurate'):
        #     print(f"ERROR")
        #     exit()
        # print(f"alpha2: {alpha2.value.T}")
        # robot.step( u2.value )
        # robot.render_plot()

        if live:
            fig.canvas.draw()
            fig.canvas.flush_events()

    return t*dt
        
robot_x = np.array([6,1.0,0,0])

xsize = 10 #20
ysize = 20 #50
posx = np.linspace( 4, 6, xsize )
posy = np.linspace( -4, 4, ysize )
xv, yv = np.meshgrid( posx, posy )

times = np.zeros((ysize, xsize))

for i in range(posx.size):
    for j in range(posy.size):
        times[j, i] = simulate(np.array([ posx[i], posy[j], 0, 0 ]))
times = np.flipud(times)
from matplotlib import cm
from matplotlib.cm import ScalarMappable
plt.ioff()
times = times/tf
my_cmap = plt.get_cmap("viridis")
sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(tf))
sm.set_array([])

fig_t, ax_t = plt.subplots()
# h = ax_t.contourf(posx, posy, times, color=my_cmap(times))
# # fig_t.colorbar(h)
# cbar = fig_t.colorbar(sm, ax=ax_t)

pos = ax_t.imshow( times, cmap='RdBu', interpolation='bilinear',extent=[4,6,-4,4], vmin=0, vmax=1.0 )
fig_t.colorbar(pos, ax=ax_t, label='Time Steps to infeasibility')
ax_t.set_ylabel("a")
ax_t.set_xlabel("b")
# print(f"")


# simulate(robot_x)



# fig_u, ax_u = plt.subplots(2)
# ax_u[0].plot(robot.Us[0,1:], "b*")
# ax_u[1].plot(robot.Us[1,1:], 'b*')
# # ax_u[2].plot(alpha2s)
# ax_u[0].set_ylabel(r'$a_x$')
# ax_u[1].set_ylabel(r'$a_y$')

# fig_alpha, ax_alpha = plt.subplots(len(obs))
# for i in range(len(obs)):
#     ax_alpha[i].plot(alpha2s[i,1:], 'b*')
#     ax_alpha[i].set_ylabel(r'$\alpha$'+f"{i}")

# fig_u.savefig('media/'+name+"control.png")
# fig_u.savefig('media/'+name+"alphas.png")

fig_t.savefig('media/'+name+'stats.png')
fig_t.savefig('media/'+name+'stats.eps')

plt.show()

