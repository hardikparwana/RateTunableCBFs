import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import time
from robot_models.obstacles import circle2D
from utils.utils import *
from robot_models.si2DJIT import *
from robot_models.unicycle2DJIT import *
from robot_models.si_mpc import SI
from robot_models.unicycle_mpc import Unicycle

dt_inner = 0.05
N = 100#100
tf =  int( N * dt_inner ) #20
outer_loop = 2
num_gd_iterations = 1
dt_outer = 0.05
H = 100#100
lr_alpha = 0.05#0.05
plot_x_lim = (-0.5,10)  
plot_y_lim = (-3,3) 

# starting point
# X_init = np.array([-0.5,-0.5,np.pi/2])
follower_Xinit = np.array([0,0,0]).reshape(-1,1)
leader_Xinit = np.array([1,0]).reshape(-1,1)
d_obs = 0.3

# input bounds
u1_max = 2
u2_max = 3


##  Define Controller ################

# si
num_constraints = 4
u = cp.Variable((2,1))
u_ref = cp.Parameter((2,1), value = np.zeros((2,1)))
A1 = cp.Parameter((num_constraints,2), value = np.zeros((num_constraints,2)))
b1 = cp.Parameter((num_constraints,1), value = np.zeros((num_constraints,1)))
delta = cp.Variable((num_constraints,1), value = np.zeros((num_constraints,1)))
objective = cp.Minimize( cp.sum_squares( u - u_ref ) + 10 * cp.sum_squares( delta[0,0] ) + 10000 * cp.sum_squares(delta[1:,:]) ) 
# factor_matrix = np.zeros((num_constraints,1)); factor_matrix[0,0] = 1
const = [A1 @ u + b1 + delta >= 0]
const += [ cp.abs( u[0,0] ) <= u1_max ]
const += [ cp.abs( u[1,0] ) <= u2_max ]
cbf_controller = cp.Problem( objective, const )
assert cbf_controller.is_dpp()
solver_args = {
            'verbose': False
        }
cbf_controller_layer = CvxpyLayer( cbf_controller, parameters=[ u_ref, A1, b1 ], variables = [u, delta] )
######################################


def initialize_tensors(follower, leader, params):
    follower.X_torch = torch.tensor( follower.X, dtype=torch.float ) 
    leader.X_torch = torch.tensor( leader.X, dtype=torch.float  )
    follower.params = torch.tensor( params, dtype=torch.float, requires_grad=True )

def compute_reward(follower, leader, params, t, dt_outer):
    
    statesF = [follower.X_torch]
    statesL = [leader.X_torch]
    reward = torch.tensor([0],dtype=torch.float)
    
    # make them tensors??
    maintain_constraints = []
    improve_constraints = []  
    global H
    for i in range(H):

        # make  control matrices for Follower
        control_ref = unicycle_nominal_input_jit( statesF[i], statesL[i]  )
        leader_xdot = SI_leader_motion(t)
        A, b = traced_unicycle2D_LF_qp_constraints_jit( statesF[i], statesL[i], leader_xdot, params[0], params[1], params[2], params[3] )
        control, deltas = cbf_controller_layer( control_ref, A, b )
        
        # Check for constraints that need to be maintained or kept
        if np.any( deltas[1:].detach().numpy() > 0.01 ):
            print(f"Error, control:{control.T}, delta:{deltas.T}")
        #     # if improve_constraints == []:
        #     #     improve_constraints = 
        #     improve_constraints = torch.cat( improve_constraints )
        #     return reward, improve_constraints, maintain_constraints, False
        # else:
        #     maintain_constraints = torch.cat( (maintain_constraints, A @ control), dim=0 )
                   
        # Get Leader state dot
        next_leader_state = update_si_state_jit( statesL[i], SI_leader_motion(t), dt_outer )
                   
        # Get next state
        next_state = update_unicycle_state_jit( statesF[i], control, dt_outer )
                
        # Save next state and compute reward
        statesF.append( next_state )
        statesL.append( next_leader_state )
        reward = reward + traced_unicycle_fov_compute_reward_jit( statesF[i+1], statesL[i+1] ) 
        
        t = t + dt_outer      
        
    return reward, improve_constraints, maintain_constraints, True

def constrained_update( objective, maintain_constraints, improve_constraints, params ) :
    
    num_params = len(params)
    d = cp.Variable((num_params,1))
    
    # Get Performance optimal direction
    objective.sum().backward(retain_graph = True) 
    param_grad = getGrad(params, l_bound = -20.0, u_bound = 20.0 )
    objective_grad = param_grad.reshape(1,-1)
    
    # Get constraint improve direction # assume one at a time
    improve_constraint_direction = np.array([0,0,0,0]).reshape(1,-1)
    for i, constraint in enumerate( improve_constraints):
        constraint.sum().backward(retain_graph=True)
        param_grad = getGrad(params, l_bound = -20.0, u_bound = 20.0 )
        improve_constraint_direction = improve_constraint_direction +  param_grad.reshape(1,-1)
        
    # Get allowed directions
    N = len(maintain_constraints)
    if N>0:
        d_maintain = np.zeros((N,num_params))
        constraints = []
        for i, constraint in enumerate(maintain_constraints):
            constraint.sum().backward(retain_graph=True)
            param_grad = getGrad(params, l_bound = -20.0, u_bound = 20.0) 
            d_maintain[i,:] = param_grad.reshape(1,-1)[0]
            
            if constraints ==[]: 
                constraints = constraint.detach().numpy().reshape(-1,1)
            else:
                constraints = np.append( constraints, constraint.detach().numpy().reshape(-1,1), axis = 0 )       

        const = [ constraints + d_maintain @ d >= 0 ]
        const += [ cp.sum_squares( d ) <= 200 ]
        if len(improve_constraint_direction)>0:
            obj = cp.Minimize( improve_constraint_direction @ d )
        else:
            obj = cp.Minimize(  objective_grad @ d  )
        problem = cp.Problem( obj, const )    
        problem.solve( solver = cp.GUROBI )    
        if problem.status != 'optimal':
            print("Cannot Find feasible direction")
            exit()
        
        return d.value
    
    else:
        if len( improve_constraints ) > 0:
            obj = cp.Maximize( improve_constraint_direction @ d )
            return -improve_constraint_direction.reshape(-1,1)
        else:
            # print("HERE>>>>>>>>>>>>>>>>>>>")
            return -objective_grad.reshape(-1,1)
        



def simulate_scenario( movie_name = 'test.mp4', adapt = True, enforce_input_constraints = False, params = [1.0, 0.8, 0.8], plot_x_lim = (-5,5), plot_y_lim = (-5,5), offline = False, offline_iterations = 20 ):

    t = 0
    
    plt.ion()
    fig = plt.figure()
    ax = plt.axes( xlim = plot_x_lim, ylim = plot_y_lim )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect(1)
    
    follower = Unicycle( follower_Xinit, dt_inner, ax, plot = False, plot_fov = True, palpha = 0 )
    leader = SI( leader_Xinit, dt_inner, ax, nominal_plot=False )
    global obs1X, obs2X
    
    params_copy = np.copy( np.asarray(params) )
    
    i = 0
    step_rewards = []
    step_params = np.asarray(params).reshape(-1,1)
    
    initialize_tensors( follower, leader, params )
    
    offline_done = False
    global H    
    metadata = dict(title='Movie Adapt 0', artist='Matplotlib',comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    with writer.saving(fig, 'test.mp4', 100): 

        while (t < tf):

            i = i + 1
            
            if (H>2):
                H = H - 1

            if ((i % outer_loop != 0) and (not offline)) or ( offline and offline_done ): # compute control input and move the robot
                
                follower.X_torch = torch.tensor(follower.X, dtype=torch.float)
                leader.X_torch = torch.tensor(leader.X, dtype=torch.float)
            
                # get controller
                control_ref = unicycle_nominal_input_jit( follower.X_torch, leader.X_torch  )
                leader_xdot = SI_leader_motion(torch.tensor(t, dtype=torch.float))
                A, b = traced_unicycle2D_LF_qp_constraints_jit( follower.X_torch, leader.X_torch, leader_xdot, torch.tensor(params[0], dtype=torch.float), torch.tensor(params[1], dtype=torch.float), torch.tensor(params[2], dtype=torch.float), torch.tensor(params[3], dtype=torch.float) )
                control, deltas = cbf_controller_layer( control_ref, A, b )
                
                # print(f"control: {control.T}")
                follower.step( control.detach().numpy() )
                follower.render_plot()
                
                leader.step( leader_xdot.detach().numpy() )
                
                step_rewards.append( traced_unicycle_fov_compute_reward_jit( torch.tensor(follower.X_torch, dtype=torch.float), leader.X_torch).item() )
                step_params = np.append( step_params, np.asarray(params).reshape(-1,1), axis=1 )
                
                # Nominal robot
                follower.X_nominal_torch = torch.tensor(follower.X_nominal, dtype=torch.float)
                control_ref = unicycle_nominal_input_jit( follower.X_nominal_torch, leader.X_torch  )
                A, b = traced_unicycle2D_LF_qp_constraints_jit( follower.X_nominal_torch, leader.X_torch, leader_xdot, torch.tensor(params_copy[0], dtype=torch.float), torch.tensor(params_copy[1], dtype=torch.float), torch.tensor(params_copy[2], dtype=torch.float), torch.tensor(params_copy[3], dtype=torch.float) )
                control, deltas = cbf_controller_layer( control_ref, A, b )            
                follower.step_nominal( control.detach().numpy() )
                follower.render_plot_fov()
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                writer.grab_frame()
            
                t = t + dt_inner
                
            else: # update parameters with predictive framework
                
                initialize_tensors(follower, leader, params )
                
                if (not offline): 
                
                    for k in range(num_gd_iterations):
                        success = False                    
                        while not success:            
                            reward, improve_constraints, maintain_constraints, success = compute_reward(follower, leader, follower.params, torch.tensor(t, dtype=torch.float), torch.tensor(dt_outer, dtype=torch.float))
                            grads = constrained_update( reward, maintain_constraints, improve_constraints, follower.params )
                            
                            grads = np.clip( grads, -2.0, 2.0 )
                            
                            params[0] = np.clip( params[0] + lr_alpha * grads[0], 0.0, None ).item()
                            params[1] = np.clip( params[1] + lr_alpha * grads[1], 0.0, None ).item()
                            params[2] = np.clip( params[2] + lr_alpha * grads[2], 0.0, None ).item()
                            params[3] = np.clip( params[3] + lr_alpha * grads[3], 0.0, None ).item()
                            # print(f"grads: {grads.T}, params: {params}")
                else:
                    
                    for k in range( offline_iterations ):        
                        print(f"k:{k}")            
                        success = False                    
                        while not success:            
                            reward, improve_constraints, maintain_constraints, success = compute_reward(follower, leader, follower.params, torch.tensor(t, dtype=torch.float), torch.tensor(dt_outer, dtype=torch.float))
                            grads = constrained_update( reward, maintain_constraints, improve_constraints, follower.params )
                            
                            grads = np.clip( grads, -2.0, 2.0 )
                            
                            params[0] = np.clip( params[0] + lr_alpha * grads[0], 0.0, None ).item()
                            params[1] = np.clip( params[1] + lr_alpha * grads[1], 0.0, None ).item()
                            params[2] = np.clip( params[2] + lr_alpha * grads[2], 0.0, None ).item()
                            params[3] = np.clip( params[3] + lr_alpha * grads[3], 0.0, None ).item()
                            # print(f"grads: {grads.T}, params: {params}")
                            
                    offline_done = True
                    
                
    return fig, ax, follower, leader, step_rewards, step_params

            
            
# Run simulations
fig1, ax1, follower1, leader1, rewards1, params1 = simulate_scenario( movie_name = 'si_2d/figures/cs4_case1_rc.mp4', adapt=True, enforce_input_constraints=True, params = [1.0, 3.0, 3.0, 3.0], plot_x_lim = plot_x_lim, plot_y_lim = plot_y_lim, offline = False, offline_iterations=20 )            
# fig2, ax2, follower2, leader2, rewards2, params2 = simulate_scenario( movie_name = 'si_2d/figures/cs4_case2_rc.mp4', adapt=True, enforce_input_constraints=True, params = [0.5, 0.5, 0.5], plot_x_lim = plot_x_lim, plot_y_lim = plot_y_lim, offline = False, offline_iterations=20 )            
# fig3, ax3, follower3, leader3, rewards3, params3 = simulate_scenario( movie_name = 'si_2d/figures/cs4_case1_offline.mp4', adapt=True, enforce_input_constraints=True, params = [1.0, 3.0, 3.0], plot_x_lim = plot_x_lim, plot_y_lim = plot_y_lim, offline = True, offline_iterations=20 )            
# fig4, ax4, follower4, leader4, rewards4, params4 = simulate_scenario( movie_name = 'si_2d/figures/cs4_case2_offline.mp4', adapt=True, enforce_input_constraints=True, params = [0.5, 0.5, 0.5], plot_x_lim = plot_x_lim, plot_y_lim = plot_y_lim, offline = True, offline_iterations=20 )

plt.ioff()

with open('si_2d/mpc_case1.npy', 'rb') as f:
    Xs = np.load(f)
    
fig, ax = plt.subplots(1,1)
ax.set_xlim( plot_x_lim )
ax.set_ylim( plot_y_lim )

# Plot obstacles
circ = plt.Circle((obs1X[0],obs1X[1]),d_obs, linewidth = 1, edgecolor='k',facecolor='k')
ax.add_patch(circ)
circ2 = plt.Circle((obs2X[0],obs2X[1]),d_obs, linewidth = 1, edgecolor='k',facecolor='k')
ax.add_patch(circ2)

# Plot MPC solution
ax.plot(Xs[0,1:], Xs[1,1:],'r', label='MPC')

# Plot new solution

# Case 1
ax.plot(robot1.Xs_nominal[0,:], robot1.Xs_nominal[1,:], 'y', label='Nominal Case 1')
ax.plot(robot1.Xs[0,:], robot1.Xs[1,:], 'g', label='RC Case 1')
ax.plot(robot3.Xs[0,:], robot3.Xs[1,:], 'b', label='Offline Case 1')

# Case 2
ax.plot(robot2.Xs_nominal[0,:], robot2.Xs_nominal[1,:], 'y--', label='Nominal Case 2')
ax.plot(robot2.Xs[0,:], robot2.Xs[1,:], 'g--', label='RC Case 2')
ax.plot(robot4.Xs[0,:], robot4.Xs[1,:], 'b--', label='Offline Case 2')



# Show plot
ax.legend()
fig.savefig("si_2d/figures/cs4.png")
plt.show()


