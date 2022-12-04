import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator3D import *
from robot_models.SingleIntegrator6D import *
from robot_models.Unicycle2D import *


# sim parameters
dt = 0.01
d_min = 0.3
T = 100

# trust parameters
min_dist = 1.0
h_min = 1.0
alpha_der_max = 0.1

# figure
plt.ion()
fig = plt.figure()#(dpi=100)
# fig.set_size_inches(33, 15)
ax = plt.axes(projection ="3d",xlim=(-5,5),ylim=(-5,5), zlim=(-0.01,4.0))   
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1,1,4.0/10])

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

robots.append( SingleIntegrator6D(np.array([-1.5,1.5,height,0,0,0]), dt, ax, id = 0, cone_length = height/np.cos(cone_angle), cone_angle = cone_angle, mode = 'uncooperative', target=np.array([0,0,0,0,0,0]).reshape(-1,1)) )
robots.append( SingleIntegrator6D(np.array([1.5,-1.0,height,0,0,0]), dt, ax, id = 1, cone_length = height/np.cos(cone_angle), cone_angle = cone_angle, mode = 'uncooperative', target = np.array([0,0,0,0,0,0]).reshape(-1,1)) )
robots.append( SingleIntegrator3D(np.array([1,1,0]), dt, ax, id = 2, color = 'g', mode='adversary', target = 5 ) )
robots.append( SingleIntegrator3D(np.array([2,4,0]), dt, ax, id = 3, color = 'g', mode='ego', target = np.array([1,0]).reshape(-1,1) ) )
robots.append( SingleIntegrator3D(np.array([-2,-4,0]), dt, ax, id = 4, color = 'g', mode='ego', target = np.array([1,0]).reshape(-1,1) ) )
robots.append( Unicycle2D( np.array([-1,-1,0,0]), dt, ax, id = 5, color='b', mode='ego', target = np.array([1,0]).reshape(-1,1)) )
robots.append( Unicycle2D( np.array([4,2,0,np.pi]), dt, ax, id = 6, color='b', mode='ego', target = np.array([1,0]).reshape(-1,1) ) )
robots.append( Unicycle2D( np.array([-4,-2,0,np.pi/3]), dt, ax, id = 7, color='b', mode='ego', target = np.array([1,0]).reshape(-1,1) ) )

num_robots = len(robots)


### Controllers
### 1. 2 control inputs
u2 = cp.Variable((2,1))
u2_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints2  = num_robots - 1
A2 = cp.Parameter((num_constraints2,2),value=np.zeros((num_constraints2,2)))
b2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
slack_constraints2 = cp.Parameter( (num_constraints2,1), value = np.zeros((num_constraints2,1)) )
const2 = [A2 @ u2 >= b2 + slack_constraints2]
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
objective3 = cp.Minimize( cp.sum_squares( u3 - u3_ref ) )
cbf_controller3 = cp.Problem( objective3, const3 )

max_u = 4
### 3. trust computation for 2: constraints are same
uT2 = cp.Variable( (2,1) )
QT2 = cp.Parameter( (1,2), value = np.zeros((1,2)) )
constT2 = [A2 @ uT2 <= b2]
constT2 += [ cp.abs( u2[0,0] ) <= max_u ]
constT2 += [ cp.abs( u2[1,0] ) <= max_u ]
objectiveT2 = cp.Minimize( QT2 @ uT2 )
best_controllerT2 = cp.Problem( objectiveT2, constT2 )

### 3. trust computation for 3: constraints are same
uT3 = cp.Variable( (3,1) )
QT3 = cp.Parameter( (1,3), value = np.zeros((1,3)) )
constT3 = [A3 @ uT3 <= b3]
constT3 += [ cp.abs( u3[0,0] ) <= max_u ]
constT3 += [ cp.abs( u3[1,0] ) <= max_u ]
constT3 += [ cp.abs( u3[2,0] ) <= max_u ]
objectiveT3 = cp.Minimize( QT3 @ uT3 )
best_controllerT3 = cp.Problem( objectiveT3, constT3 )

# plt.show()
for t in range(T):    
    # robots[0].step(np.array([0.0,0.0,0.0,0.0,0,0]))
    # robots[2].step(np.array([1.0,0.0,0.0]))
    # robots[5].step(robots[5].nominal_input( np.array([1,1]) ))
    
    # nominal motions of agents: only to define nominal trajectory
    for i in range(num_robots):
        robots[i].U_nominal = robots[i].target    
        robots[i].step_nominal(robots[i].U_nominal)
        robots[i].render_plot_nominal()
    
    # actual agents: nominal input + make constraint matrix
    for i in range(num_robots):
        
        const_index = 0
        
        if robots[i].mode == 'uncooperative': # only velocity commands: do nominal directly
            robots[i].U_ref = robots[i].target;
            
        elif robots[i].mode == 'adversary': # only 3D single integrators
            V, dV_dxi, dV_dxj = robots[i].lyapunov( robots[robots[i].target].X, robots[robots[i].target].type  )
            robots[i].U_ref = - 1.0 * dV_dxi.T / np.linalg.norm( dV_dxi )
        
        elif robots[0].mode == 'ego':  # do our controller
            # get reference control input
            robots[i].u_ref = robots[i].nominal_input( robots[i].X_nominal, robots[i].type  )
            
            # get constraint matrix            
            for j in range(num_robots):
                if j==i:
                    continue
                
                # safety constraint
                h, dh_dxi, dh_dxj = robots[i].agent_barrier(robots[j], d_min)
                robots[i].robot_h[0,j] = h
                
                # Inequality constraint
                robots[i].A[const_index,:] = dh_dxi @ robots[i].g()
                robots[i].b[const_index] = - dh_dxi @ robots[i].f() - dh_dxj @ ( robots[j].f() + robots[j].g() @ robots[j].U ) - robots[i].alpha[0,j] * h
                
                # Best case LP objective
                robots[i].agent_objective[j] = dh_dxi @ robots[i].g()
                
                const_index += 1
                
    
    # get trust factor
    for i in range(num_robots):
        
        if robots[i].mode == 'ego':
            if robots[i].U.shape[0] == 2:
                A2.value = robots[i].A
                b2.value = robots[i].b
                
                for j in range(num_robots):
                    if j==i:
                        continue
                    QT2.value = robots[i].agent_objective                
                    best_controllerT2.solve( solver=cp.GUROBI )
                    
                    h, dh_dxi, dh_dxj = robots[i].agent_barrier(robots[j], d_min)                        
                    A = dh_dxj @ robots[j].g()
                    b = -robots[i].alpha[0,j] * h  - dh_dxi @ ( robots[i].f() + robots[i].g() @ uT2.value ) - dh_dxj @ robots[j].f() #- dh_dxi @ robots[j].U                    
                    robots[i].trust[0,j] = compute_trust( A, b, robots[j].f() + robots[j].g() @ robots[j].U, robots[j].x_dot_nominal, h, min_dist, h_min )  
                    robots[i].alpha[0,j] = robots[i].robot_alpha[0,j] + alpha_der_max * robots[i].trust_robot[0,j]
    
    
    # implement control input
    for i in range(num_robots):
        
        if robots[i].mode == 'uncooperative' or robots[i].mode == 'adversarial':
            robots[i].step( robots[i].U_ref )
        elif robots[i].mode == 'ego':
            if robots[i].U.shape[0] == 2:
                A2.value = robots[i].A2
                b2.value = robots[i].b2
                cbf_controller2.solve(solver=cp.GUROBI)
                if cbf_controller2.status != 'optimal':
                    print("Error: QP not feasible")
                robots[i].step( u2.value )
            elif robots[i].U.shape[0] == 3:
                A3.value = robots[i].A3
                b3.value = robots[i].b3
                cbf_controller3.solve(solver=cp.GUROBI)
                if cbf_controller3.status != 'optimal':
                    print("Error: QP not feasible")
                robots[i].step( u3.value )
        robots[i].render_plot()
            
    # store data for plotting
    for i in range(num_robots):
        if robots.mode == 'ego':
            robots[j].alphas = np.append( robots[j].alphas, robots[j].alpha, axis=0 )
            robots[j].trusts = np.append( robots[j].trusts, robots[j].trust, axis=0 )
            robots[j].hs = np.append( robots[j].hs, robots[j].h, axis=0 )           
    
    fig.canvas.draw()
    fig.canvas.flush_events()