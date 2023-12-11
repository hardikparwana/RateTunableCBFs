import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator3D import *
from robot_models.Surveillance import *
from robot_models.Unicycle2D import *
from robot_models.DoubleIntegrator3D import *
from trust_utils import *
from matplotlib.animation import FFMpegWriter

# sim parameters
dt = 0.05
tf = 8
d_min = 0.3
T = int( tf/dt )
model_trust = True
fixed_parameter = True

# trust parameters
min_dist = 0.6#0.4#1.0 # important. set separately for double integrator
h_min = 1.0#0.6  # important. 
# min_dist and h_min are better tuned in 1-1 scenario. Then use the same in multi-agent scenario. usually works fine. each type of agent and barrier function might need different type of tuning
alpha_der_max = 0.05#0.5#0.1 # very important parameter
alpha = 0.8  # very important parameter

# figure
plt.ion()
fig = plt.figure()#(dpi=100)
# fig.set_size_inches(33, 15)
ax = plt.axes(projection ="3d",xlim=(-5,5),ylim=(-5,5), zlim=(-0.01,2.0))   
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# plt.axis('off')
# ax.set_zticks([])
ax.set_box_aspect([1,1,2.0/10])
# ax.view_init(90, 0)
# Set ground plane
length = 5
height = -0.1
x = [-length,-length,length,length]
y = [-length,length,length,-length]
z = [-height,height,height,height]
verts = [list(zip(x,y,z))]
# ax.add_collection3d(Poly3DCollection(verts,facecolor='gray', alpha=0.5))

robots = []

cone_angle = np.pi/6       
height = 2.0

nominal_plot = False
robot_plot = True
num_robots = 6
num_constraints = num_robots - 1


# Adversary
robots.append( SingleIntegrator3D(np.array([-3,1,0]), dt, ax, id = 0, nominal_plot = nominal_plot, alpha = alpha, grounded = True, color = 'k', mode='adversary', target = 1, num_robots = num_robots, num_constraints = num_constraints, plot = robot_plot ) )

# Ego
robots.append( Unicycle2D( np.array([0,-2,0, np.pi/2]), dt, ax, id = 1, nominal_plot = nominal_plot, color='b', alpha = alpha, mode='ego', target = np.array([1.2,0]).reshape(-1,1), num_robots = num_robots, num_constraints = num_constraints, plot = robot_plot) )
robots.append( Unicycle2D( np.array([-1.5,-3,0, np.pi/2]), dt, ax, id = 2, nominal_plot = nominal_plot, color='b', alpha = alpha, mode='ego', target = np.array([1.2,0]).reshape(-1,1), num_robots = num_robots, num_constraints = num_constraints, plot = robot_plot ) )

# Noncooperative
robots.append( Surveillance(np.array([4,2,height,0,0,0]), dt, ax, id = 3, cone_length = height/np.cos(cone_angle), cone_angle = cone_angle, mode = 'uncooperative', target = np.array([-1,0,0,0,0,0]).reshape(-1,1), num_robots = num_robots, num_constraints = num_constraints, plot = robot_plot) )

# higher order ego
robots.append( DoubleIntegrator3D( np.array([-4,-2,0,0,0,0]), dt, ax, nominal_plot = nominal_plot, id = 4, color='g', alpha = alpha, mode='ego', target = np.array([0.6,0.6,0.1]).reshape(-1,1), num_robots = num_robots, num_constraints = num_constraints, plot = robot_plot  ) )
robots.append( DoubleIntegrator3D( np.array([-4,2.2,0.5,0,0,0]), dt, ax, nominal_plot = nominal_plot, id = 4, color='g', alpha = alpha, mode='ego', target = np.array([0.6,0.0,0.0]).reshape(-1,1), num_robots = num_robots, num_constraints = num_constraints, plot = robot_plot  ) )

# plt.ioff()
# plt.show()

# Fixed parameter
robots_fixed = []
nominal_plot = False
robots_fixed.append( SingleIntegrator3D(np.array([-3,1,0]), dt, ax, id = 0, nominal_plot = nominal_plot, alpha = alpha, grounded = True, color = 'k', mode='adversary', target = 1, num_robots = num_robots, num_constraints = num_constraints, plot=nominal_plot ) )
robots_fixed.append( Unicycle2D( np.array([0,-2,0, np.pi/2]), dt, ax, id = 1, nominal_plot = nominal_plot, color='b', alpha = alpha, mode='ego', target = np.array([1.2,0]).reshape(-1,1), num_robots = num_robots, num_constraints = num_constraints, plot=nominal_plot) )
robots_fixed.append( Unicycle2D( np.array([-1.5,-3,0, np.pi/2]), dt, ax, id = 2, nominal_plot = nominal_plot, color='b', alpha = alpha, mode='ego', target = np.array([1.2,0]).reshape(-1,1), num_robots = num_robots, num_constraints = num_constraints, plot=nominal_plot ) )
robots_fixed.append( Surveillance(np.array([4,2,height,0,0,0]), dt, ax, id = 3, cone_length = height/np.cos(cone_angle), cone_angle = cone_angle, mode = 'uncooperative', target = np.array([-1,0,0,0,0,0]).reshape(-1,1), num_robots = num_robots, num_constraints = num_constraints, plot=nominal_plot) )
robots_fixed.append( DoubleIntegrator3D( np.array([-4,-2,0,0,0,0]), dt, ax, nominal_plot = nominal_plot, id = 4, color='g', alpha = alpha, mode='ego', target = np.array([0.6,0.6,0.1]).reshape(-1,1), num_robots = num_robots, num_constraints = num_constraints, plot=nominal_plot  ) )
robots_fixed.append( DoubleIntegrator3D( np.array([-4,2.2,0.5,0,0,0]), dt, ax, nominal_plot = nominal_plot, id = 4, color='g', alpha = alpha, mode='ego', target = np.array([0.6,0.0,0.0]).reshape(-1,1), num_robots = num_robots, num_constraints = num_constraints, plot=nominal_plot  ) )

### Controllers
max_u = 5 
### 1. 2 control inputs
u2 = cp.Variable((2,1))
u2_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints2  = num_robots - 1
A2 = cp.Parameter((num_constraints2,2),value=np.zeros((num_constraints2,2)))
b2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
slack_constraints2 = cp.Parameter( (num_constraints2,1), value = np.zeros((num_constraints2,1)) )
const2 = [A2 @ u2 >= b2 + slack_constraints2]
const2 += [ cp.abs( u2[0,0] ) <= max_u ]
const2 += [ cp.abs( u2[1,0] ) <= max_u ]
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref  ) )
cbf_controller2 = cp.Problem( objective2, const2 )

### 2. 3 control inputs
u3 = cp.Variable((3,1))
u3_ref = cp.Parameter((3,1),value = np.zeros((3,1)) )
num_constraints3  = num_robots - 1
A3 = cp.Parameter((num_constraints3,3),value=np.zeros((num_constraints3,3)))
b3 = cp.Parameter((num_constraints3,1),value=np.zeros((num_constraints3,1)))
slack_constraints3 = cp.Parameter( (num_constraints3,1), value = np.zeros((num_constraints3,1)) )
const3 = [A3 @ u3 >= b3 + slack_constraints3]
const3 += [ cp.abs( u3[0,0] ) <= max_u ]
const3 += [ cp.abs( u3[1,0] ) <= max_u ]
const3 += [ cp.abs( u3[2,0] ) <= max_u ]
objective3 = cp.Minimize( cp.sum_squares( u3 - u3_ref ) )
cbf_controller3 = cp.Problem( objective3, const3 )

### 3. trust computation for 2: constraints are same
uT2 = cp.Variable( (2,1) )
QT2 = cp.Parameter( (1,2), value = np.zeros((1,2)) )
constT2 = [A2 @ uT2 >= b2]
constT2 += [ cp.abs( uT2[0,0] ) <= max_u ]
constT2 += [ cp.abs( uT2[1,0] ) <= max_u ]
objectiveT2 = cp.Maximize( QT2 @ uT2 )
best_controllerT2 = cp.Problem( objectiveT2, constT2 )

### 3. trust computation for 3: constraints are same
uT3 = cp.Variable( (3,1) )
QT3 = cp.Parameter( (1,3), value = np.zeros((1,3)) )
constT3 = [A3 @ uT3 >= b3]
constT3 += [ cp.abs( uT3[0,0] ) <= max_u ]
constT3 += [ cp.abs( uT3[1,0] ) <= max_u ]
constT3 += [ cp.abs( uT3[2,0] ) <= max_u ]
objectiveT3 = cp.Maximize( QT3 @ uT3 )
best_controllerT3 = cp.Problem( objectiveT3, constT3 )

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=12, metadata=metadata)

with writer.saving(fig, 'scenario1/take1_adaptive.mp4', 100): 

    for t in range(T):    
        # nominal motions of agents: only to define nominal trajectory
        for i in range(num_robots):
            
            if robots[i].mode=='uncooperative':
                robots[i].U_nominal = robots[i].target    
            elif robots[i].mode == 'adversary':
                V, dV_dxi, dV_dxj = robots[i].lyapunov_nominal( robots[robots[i].target].X_nominal, robots[robots[i].target].type )
                robots[i].U_nominal = -1.0*dV_dxi.T/np.linalg.norm(dV_dxi)
            elif robots[i].mode == 'ego':
                robots[i].U_nominal = robots[i].target
            robots[i].step_nominal(robots[i].U_nominal)
            robots[i].x_dot_nominal = robots[i].f() + robots[i].g() @ np.copy(robots[i].U)
            
        # Fixed parameter
        if fixed_parameter:
            for i in range(num_robots):
                
                if robots_fixed[i].mode=='uncooperative':
                    robots_fixed[i].U_nominal = robots_fixed[i].target    
                elif robots_fixed[i].mode == 'adversary':
                    V, dV_dxi, dV_dxj = robots_fixed[i].lyapunov_nominal( robots_fixed[robots_fixed[i].target].X_nominal, robots_fixed[robots_fixed[i].target].type )
                    robots_fixed[i].U_nominal = -1.0*dV_dxi.T/np.linalg.norm(dV_dxi)
                elif robots_fixed[i].mode == 'ego':
                    robots_fixed[i].U_nominal = robots_fixed[i].target
                robots_fixed[i].step_nominal(robots_fixed[i].U_nominal)
                robots_fixed[i].x_dot_nominal = robots_fixed[i].f() + robots_fixed[i].g() @ np.copy(robots_fixed[i].U)
        

        # actual agents: nominal input + make constraint matrix
        for i in range(num_robots):
            
            const_index = 0
            
            if robots[i].mode == 'uncooperative': # only velocity commands: do nominal directly
                robots[i].U_ref = robots[i].target;
                
            elif robots[i].mode == 'adversary': # only 3D single integrators
                V, dV_dxi, dV_dxj = robots[i].lyapunov( robots[robots[i].target].X, robots[robots[i].target].type  )
                robots[i].U_ref = - 1.0 * dV_dxi.T / np.linalg.norm( dV_dxi )
            
            elif robots[i].mode == 'ego':  # do our controller
                # get reference control input
                robots[i].U_ref = robots[i].nominal_input( robots[i].X_nominal, robots[i].type  )#/3
                
                # get constraint matrix            
                for j in range(num_robots):
                    if j==i:
                        continue
                    
                    # safety constraint
                    h, dh_dxi, dh_dxj = robots[i].agent_barrier(robots[j], d_min)
                    
                    # Inequality constraint
                    robots[i].A[const_index,:] = dh_dxi @ robots[i].g()
                    robots[i].b[const_index] = - dh_dxi @ robots[i].f() - robots[i].alpha[0,j] * h - dh_dxj @ ( robots[j].f() + robots[j].g() @ robots[j].U )
                    
                    # Best case LP objective
                    robots[i].agent_objective[j] = dh_dxi @ robots[i].g() # h positive
                    
                    const_index += 1
                    
                    #Plot
                    robots[i].h[0,j] = h
                    
        if fixed_parameter:
            # actual agents: nominal input + make constraint matrix
            for i in range(num_robots):
                
                const_index = 0        
                if robots_fixed[i].mode == 'uncooperative': # only velocity commands: do nominal directly
                    robots_fixed[i].U_ref = robots_fixed[i].target;            
                elif robots_fixed[i].mode == 'adversary': # only 3D single integrators
                    V, dV_dxi, dV_dxj = robots_fixed[i].lyapunov( robots_fixed[robots_fixed[i].target].X, robots_fixed[robots_fixed[i].target].type  )
                    robots_fixed[i].U_ref = - 1.0 * dV_dxi.T / np.linalg.norm( dV_dxi )        
                elif robots_fixed[i].mode == 'ego':  # do our controller
                    # get reference control input
                    robots_fixed[i].U_ref = robots_fixed[i].nominal_input( robots_fixed[i].X_nominal, robots_fixed[i].type  )#/3            
                    # get constraint matrix            
                    for j in range(num_robots):
                        if j==i:
                            continue                
                        # safety constraint
                        h, dh_dxi, dh_dxj = robots_fixed[i].agent_barrier(robots_fixed[j], d_min)                
                        # Inequality constraint
                        robots_fixed[i].A[const_index,:] = dh_dxi @ robots_fixed[i].g()
                        robots_fixed[i].b[const_index] = - dh_dxi @ robots_fixed[i].f() - robots_fixed[i].alpha[0,j] * h - dh_dxj @ ( robots_fixed[j].f() + robots_fixed[j].g() @ robots_fixed[j].U )    
                        # Best case LP objective
                        robots_fixed[i].agent_objective[j] = dh_dxi @ robots_fixed[i].g() # h positive                
                        const_index += 1                
                        #Plot
                        robots_fixed[i].h[0,j] = h
                    
        
        # get trust factor
        if model_trust:
            for i in range(num_robots):
                
                if robots[i].mode == 'ego':
                    if robots[i].U.shape[0] == 2:
                        A2.value = robots[i].A
                        b2.value = robots[i].b
                        
                        for j in range(num_robots):
                            if j==i:
                                continue
                            QT2.value = robots[i].agent_objective[j]                
                            best_controllerT2.solve( solver=cp.GUROBI, reoptimize=True )
                            if best_controllerT2.status!='optimal':
                                print(f"i:{i}, Best Case LP infeasible")
                                exit()
                            
                            
                            robots[i].trust_param_update( robots[j], j, d_min, uT2.value, min_dist, h_min, alpha_der_max, dt )
                    elif robots[i].U.shape[0] == 3:
                        A3.value = robots[i].A
                        b3.value = robots[i].b
                        
                        for j in range(num_robots):
                            if j==i:
                                continue
                            QT3.value = robots[i].agent_objective[j]                
                            best_controllerT3.solve( solver=cp.GUROBI, reoptimize=True )
                            if best_controllerT3.status!='optimal':
                                print(f"i:{i}, Best Case LP infeasible")
                                exit()
                            
                            robots[i].trust_param_update( robots[j], j, d_min, uT3.value, min_dist, h_min, alpha_der_max, dt )
                            
        # implement control input
        for i in range(num_robots):
            
            if robots[i].mode == 'uncooperative' or robots[i].mode == 'adversary':
                robots[i].step( robots[i].U_ref )
            elif robots[i].mode == 'ego':
                if robots[i].U.shape[0] == 2:
                    u2_ref.value = robots[i].U_ref
                    A2.value = robots[i].A
                    b2.value = robots[i].b
                    cbf_controller2.solve(solver=cp.GUROBI, reoptimize=True)
                    if cbf_controller2.status != 'optimal':
                        print("Error: QP not feasible")
                    robots[i].step( u2.value )
                elif robots[i].U.shape[0] == 3:
                    u3_ref.value = robots[i].U_ref
                    A3.value = robots[i].A
                    b3.value = robots[i].b
                    cbf_controller3.solve(solver=cp.GUROBI, reoptimize=True)
                    if cbf_controller3.status != 'optimal':
                        print("Error: QP not feasible")
                    robots[i].step( u3.value )
            if i==3:
                GX = np.copy(robots[3].X)
                radii = 0
                if robots[4].X[2,0] < GX[2,0]: # lower height than the 
                    radii =  (GX[2,0]-robots[4].X[2,0]) * np.tan(robots[3].cone_angle)
                    GX[2,0] = robots[4].X[2,0]                
                h4 = np.linalg.norm( robots[4].X[0:3] - GX[0:3] )**2 - (d_min + radii)**2
                
                GX = np.copy(robots[3].X)
                radii = 0
                if robots[5].X[2,0] < GX[2,0]: # lower height than the 
                    radii =  (GX[2,0]-robots[5].X[2,0]) * np.tan(robots[3].cone_angle)
                    GX[2,0] = robots[5].X[2,0]                
                h5 = np.linalg.norm( robots[5].X[0:3] - GX[0:3] )**2 - (d_min + radii)**2
                
                if h4<h5:
                    robots[i].set_agent_height( robots[4].X[2,0] )
                else:
                    robots[i].set_agent_height( robots[5].X[2,0] )
                    
                    
            robots[i].render_plot()
            
            
        if fixed_parameter:
            # implement control input
            for i in range(num_robots):
                
                if robots_fixed[i].mode == 'uncooperative' or robots_fixed[i].mode == 'adversary':
                    robots_fixed[i].step( robots_fixed[i].U_ref )
                elif robots_fixed[i].mode == 'ego':
                    if robots_fixed[i].U.shape[0] == 2:
                        u2_ref.value = robots_fixed[i].U_ref
                        A2.value = robots_fixed[i].A
                        b2.value = robots_fixed[i].b
                        cbf_controller2.solve(solver=cp.GUROBI, reoptimize=True)
                        if cbf_controller2.status != 'optimal':
                            print("Error: QP not feasible")
                        robots_fixed[i].step( u2.value )
                    elif robots_fixed[i].U.shape[0] == 3:
                        u3_ref.value = robots_fixed[i].U_ref
                        A3.value = robots_fixed[i].A
                        b3.value = robots_fixed[i].b
                        cbf_controller3.solve(solver=cp.GUROBI, reoptimize=True)
                        if cbf_controller3.status != 'optimal':
                            print("Error: QP not feasible")
                        robots_fixed[i].step( u3.value )
           
        fig.canvas.draw()
        fig.canvas.flush_events()
        writer.grab_frame()
    
# Plots
plt.ioff()
# barriers: ego agent 1 id 1
figure1, axis1 = plt.subplots(1, 1)
idx = 1
size = np.shape(robots[idx].hs)[0]
indexes = np.linspace(0,tf,size)
axis1.plot( indexes, robots[idx].hs[:,0], label=r'$h_{21}$' )
axis1.plot( indexes, robots[idx].hs[:,2], label=r'$h_{23}$' )
axis1.plot( indexes, robots[idx].hs[:,3], label=r'$h_{24}$' )
axis1.plot( indexes, robots[idx].hs[:,4], label=r'$h_{25}$' )
axis1.plot( indexes, robots[idx].hs[:,5], label=r'$h_{26}$' )
axis1.set_ylim(-1,100)
axis1.legend()

figure2, axis2 = plt.subplots(1, 1)
idx = 2
size = np.shape(robots[idx].hs)[0]
indexes = np.linspace(0,tf,size)
axis2.plot( indexes, robots[idx].hs[:,0], label=r'$h_{31}$' )
axis2.plot( indexes, robots[idx].hs[:,1], label=r'$h_{32}$' )
axis2.plot( indexes, robots[idx].hs[:,3], label=r'$h_{34}$' )
axis2.plot( indexes, robots[idx].hs[:,4], label=r'$h_{35}$' )
axis2.plot( indexes, robots[idx].hs[:,5], label=r'$h_{36}$' )
axis2.set_ylim(-1,100)
axis2.legend()

figure3, axis3 = plt.subplots(1, 1)
idx = 4
size = np.shape(robots[idx].hs)[0]
indexes = np.linspace(0,tf,size)
axis3.plot( indexes, robots[idx].hs[:,0], label=r'$h_{51}$' )
axis3.plot( indexes, robots[idx].hs[:,1], label=r'$h_{51}$' )
axis3.plot( indexes, robots[idx].hs[:,2], label=r'$h_{53}$' )
axis3.plot( indexes, robots[idx].hs[:,3], label=r'$h_{54}$' )
axis3.plot( indexes, robots[idx].hs[:,5], label=r'$h_{56}$' )
axis3.set_ylim(-1,100)
axis3.legend()

figure4, axis4 = plt.subplots(1, 1)
idx = 5
size = np.shape(robots[idx].hs)[0]
indexes = np.linspace(0,tf,size)
axis4.plot( indexes, robots[idx].hs[:,0], label='h_{61}' )
axis4.plot( indexes, robots[idx].hs[:,1], label='h_{61}' )
axis4.plot( indexes, robots[idx].hs[:,2], label='h_{63}' )
axis4.plot( indexes, robots[idx].hs[:,3], label='h_{64}' )
axis4.plot( indexes, robots[idx].hs[:,4], label='h_{65}' )
axis4.set_ylim(-1,100)
axis4.legend()

# alphas
figure5, axis5 = plt.subplots(1, 1)
idx = 1
size = np.shape(robots[idx].alphas)[0]
indexes = np.linspace(0,tf,size)
axis5.plot( indexes, robots[idx].alphas[:,0], label=r'$\alpha_{21}$' )
axis5.plot( indexes, robots[idx].alphas[:,2], label=r'$\alpha_{23}$' )
axis5.plot( indexes, robots[idx].alphas[:,3], label=r'$\alpha_{24}$' )
axis5.plot( indexes, robots[idx].alphas[:,4], label=r'$\alpha_{25}$' )
axis5.plot( indexes, robots[idx].alphas[:,5], label=r'$\alpha_{26}$' )
axis5.legend()

figure6, axis6 = plt.subplots(1, 1)
idx = 2
size = np.shape(robots[idx].alphas)[0]
indexes = np.linspace(0,tf,size)
axis6.plot( indexes, robots[idx].alphas[:,0], label=r'$\alpha_{31}$' )
axis6.plot( indexes, robots[idx].alphas[:,1], label=r'$\alpha_{32}$' )
axis6.plot( indexes, robots[idx].alphas[:,3], label=r'$\alpha_{34}$' )
axis6.plot( indexes, robots[idx].alphas[:,4], label=r'$\alpha_{35}$' )
axis6.plot( indexes, robots[idx].alphas[:,5], label=r'$\alpha_{36}$' )
axis6.legend()

figure7, axis7 = plt.subplots(1, 1)
idx = 4
size = np.shape(robots[idx].alphas)[0]
indexes = np.linspace(0,tf,size)
axis7.plot( indexes, robots[idx].alphas[:,0], label=r'$\alpha_{51}$' )
axis7.plot( indexes, robots[idx].alphas[:,1], label=r'$\alpha_{51}$' )
axis7.plot( indexes, robots[idx].alphas[:,2], label=r'$\alpha_{53}$' )
axis7.plot( indexes, robots[idx].alphas[:,3], label=r'$\alpha_{54}$' )
axis7.plot( indexes, robots[idx].alphas[:,5], label=r'$\alpha_{56}$' )
axis7.legend()

figure8, axis8 = plt.subplots(1, 1)
idx = 5
size = np.shape(robots[idx].alphas)[0]
indexes = np.linspace(0,tf,size)
axis8.plot( indexes, robots[idx].alphas[:,0], label=r'$\alpha_{61}$' )
axis8.plot( indexes, robots[idx].alphas[:,1], label=r'$\alpha_{61}$' )
axis8.plot( indexes, robots[idx].alphas[:,2], label=r'$\alpha_{63}$' )
axis8.plot( indexes, robots[idx].alphas[:,3], label=r'$\alpha_{64}$' )
axis8.plot( indexes, robots[idx].alphas[:,4], label=r'$\alpha_{65}$' )
axis8.legend()

# Trajectory Plot

figure9, axis9 = plt.subplots(1, 1)
axis9 = plt.axes(projection ="3d",xlim=(-4.5,6),ylim=(-3,5), zlim=(-0.01,2.0))   
size = np.shape(robots[0].Xs)[1]-1
indexes = np.linspace(0,tf,size)

indexes = -1 + 2*indexes/indexes[-1]
indexes = indexes/np.max(indexes)

cc = np.tan(np.asarray(indexes))
im1 = axis9.scatter( robots[0].Xs[0,1:], robots[0].Xs[1,1:], robots[0].Xs[2,1:],c=cc, cmap = 'CMRmap' )
im2 = axis9.scatter( robots[1].Xs[0,1:], robots[1].Xs[1,1:], robots[1].Xs[2,1:], c=cc )
axis9.scatter( robots[2].Xs[0,1:], robots[2].Xs[1,1:], robots[2].Xs[2,1:], c=cc )
axis9.scatter( robots[3].Xs[0,1:], robots[3].Xs[1,1:], robots[3].Xs[2,1:], c=cc, cmap = 'CMRmap' )
axis9.scatter( robots[4].Xs[0,1:], robots[4].Xs[1,1:], robots[4].Xs[2,1:], c=cc )
axis9.scatter( robots[5].Xs[0,1:], robots[5].Xs[1,1:], robots[5].Xs[2,1:], c=cc )

robots.append( Surveillance(np.array([4,2,height,0,0,0]), dt, axis9, id = 3, cone_length = height/np.cos(cone_angle), cone_angle = cone_angle, mode = 'uncooperative', target = np.array([-1,0,0,0,0,0]).reshape(-1,1), num_robots = num_robots, num_constraints = num_constraints) )

figure9.colorbar(im1, ax=axis9)
figure9.colorbar(im2, ax=axis9)

if fixed_parameter:
    axis9.scatter( robots_fixed[0].Xs[0,1:], robots_fixed[0].Xs[1,1:], robots_fixed[0].Xs[2,1:],c=cc, cmap = 'CMRmap', alpha=0.1 )
    axis9.scatter( robots_fixed[1].Xs[0,1:], robots_fixed[1].Xs[1,1:], robots_fixed[1].Xs[2,1:], c=cc, alpha=0.1 )
    axis9.scatter( robots_fixed[2].Xs[0,1:], robots_fixed[2].Xs[1,1:], robots_fixed[2].Xs[2,1:], c=cc, alpha=0.1 )
    axis9.scatter( robots_fixed[3].Xs[0,1:], robots_fixed[3].Xs[1,1:], robots_fixed[3].Xs[2,1:], c=cc, cmap = 'CMRmap', alpha=0.1 )
    axis9.scatter( robots_fixed[4].Xs[0,1:], robots_fixed[4].Xs[1,1:], robots_fixed[4].Xs[2,1:], c=cc, alpha=0.1 )
    axis9.scatter( robots_fixed[5].Xs[0,1:], robots_fixed[5].Xs[1,1:], robots_fixed[5].Xs[2,1:], c=cc, alpha=0.1 )

# if save_plot:
#     figure4.savefig("trajectory.eps", dpi=50, rasterized=True)
#     figure4.savefig("trajectory.png")

plt.show()



