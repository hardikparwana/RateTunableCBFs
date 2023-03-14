import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import time
from robot_models.obstacles import circle2D
from utils.utils import *
from robot_models.bicycle2DJIT import *
from robot_models.bicycle_2d import Bicycle_2d

# import warnings
# warnings.filterwarnings("error")

dt_inner = 0.01
tf = 40
# N = int( tf/dt_inner )
# N = 100#100
# tf =  int( N * dt_inner ) #20
outer_loop = 2#0000000
num_gd_iterations = 1
dt_outer = 0.01
H = 20#100
lr_alpha = 0.1#0.05
plot_x_lim = (-1.0,3.5)  
# plot_y_lim = (-0.8,3) 
plot_y_lim = (-2.6,3)

# starting point
# X_init = np.array([-0.5,-0.5,np.pi/2])
d_obs = 0.3
X_init = np.array([0.3,-1.0, np.pi/4, 0.1 ]) #-0.3,-0.5
goalX = np.array([1.25,2.5]).reshape(-1,1) #2.5,2.0
obs1X = [0.7, 0.7]
obs2X = [1.5, 1.9]

# X_init = np.array([-0.5,-2.5, 0, 0.1, 0.1, 0, 0.1 ])
# goalX = np.array([0.9, 0.9]).reshape(-1,1)
# goalX = np.array([1.5, 2.4]).reshape(-1,1)
# obs1X = [0.5, 0.5]
# obs2X = [1.5, 1.9]

# input bounds
u1_max = 5
u2_max = 5


##  Define Controller ################

# si
num_constraints = 3 + 2
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
            'verbose': False,
            'max_iters': 100000000
        }
cbf_controller_layer = CvxpyLayer( cbf_controller, parameters=[ u_ref, A1, b1 ], variables = [u, delta] )
######################################


def initialize_tensors(robot, obs1, obs2, params):
    robot.X_torch = torch.tensor( robot.X, requires_grad = True, dtype=torch.float )
    robot.goal_torch = torch.tensor( goalX, dtype=torch.float ).reshape(-1,1)    
    obs1.X_torch = torch.tensor(obs1.X, dtype=torch.float).reshape(-1,1)
    obs2.X_torch = torch.tensor(obs2.X, dtype=torch.float).reshape(-1,1)
    robot.params = torch.tensor( params, dtype=torch.float, requires_grad=True )

def compute_reward(robot, obs1, obs2, params, dt_outer):
    
    states = [robot.X_torch]
    reward = torch.tensor([0],dtype=torch.float)
    
    # make them tensors??
    maintain_constraints = []
    improve_constraints = []  
    global H
    for i in range(H):
        # print(f"i:{i}")
        # print(f"FROM loop states:{states[i].T}")
        # make  control matrices
        # control_ref = torch.tensor([0,0], dtype=torch.float).reshape(-1,1)
        control_ref = bicycle2D_nominal_input_jit( states[i], robot.goal_torch )
        A, b = traced_bicycle2D_qp_constraints_jit( states[i], robot.goal_torch, obs1.X_torch, obs2.X_torch, params[0], params[1], params[2] )
        # print(f"A:{A}, b:{b}, u_ref:{control_ref}")
        control, deltas = cbf_controller_layer( control_ref, A, b ) #if ( (torch.abs(control[0,0]) > 11) or (torch.abs(control[1,0])>11) ): # print("wrong input: issue with solver")
        
        if ( np.abs(control[0,0].detach().numpy())>1.1*u1_max ) or ( np.abs(control[1,0].detach().numpy())>1.1*u2_max ):
            print("solved Inaccurate")
            # plt.ioff()
            # plt.show()
            # plt.ion()
            A, b = bicycle2D_qp_constraints_jit( states[i], robot.goal_torch, obs1.X_torch, obs2.X_torch, params[0], params[1], params[2], params[3] )
            
            return reward, improve_constraints, maintain_constraints, True
        
        
        # Check for constraints that need to be maintained or kept
        if np.any( deltas[1:].detach().numpy() > 0.1): #01 ):
            print(f"Error, infeasible at i:{i}, control:{control.T}, delta:{deltas.T}")
            # improve_constraints.append( -b[0] )
            if deltas[1,0].detach().numpy() > 0.1:
                improve_constraints.append( -b[1] )
            if deltas[2,0].detach().numpy() > 0.1:
                improve_constraints.append( -b[2] )
            if deltas[3,0].detach().numpy() > 0.1:
                improve_constraints.append( -b[3] )
            if deltas[4,0].detach().numpy() > 0.1:
                improve_constraints.append( -b[4] )
            # improve_constraints = []         
            return reward, improve_constraints, maintain_constraints, False
        else:
            temp = A @ control + b + deltas
            if torch.abs(temp[0])<1.0:
                maintain_constraints.append(temp[0] + 0.01)
            if torch.abs(temp[1])<1.0:
                maintain_constraints.append(temp[1] + 0.01)
            if torch.abs(temp[2])<1.0:
                maintain_constraints.append(temp[2] + 0.01)
            # maintain_constraints = []
                   
        # Get next state
        next_state = update_bicycle_state_jit( states[i], control, dt_outer )
        # print(f"FROM loop: next_state:{next_state.T}, control:{control.T}")
                
        # Save next state and compute reward
        states.append( next_state )
        reward = reward + traced_bicycle_compute_reward_jit( states[i+1], robot.goal_torch, control )       
        
    return reward, improve_constraints, maintain_constraints, True

def constrained_update( objective, maintain_constraints, improve_constraints, params ) :
    
    num_params = params.shape[0]
    d = cp.Variable((num_params,1))
    
    # Get Performance optimal direction
    objective.sum().backward(retain_graph = True) 
    param_grad = getGrad(params, l_bound = -20.0, u_bound = 20.0 )
    objective_grad = param_grad.reshape(1,-1)
    
    # Get constraint improve direction # assume one at a time
    improve_constraint_direction = np.zeros( num_params ).reshape(1,-1)
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
            obj = cp.Minimize( improve_constraint_direction @ d ) # does not do anything here
            return -improve_constraint_direction.reshape(-1,1)
        else:
            # print("HERE>>>>>>>>>>>>>>>>>>>")
            return -objective_grad.reshape(-1,1)
        



def simulate_scenario( movie_name = 'test.mp4', adapt = True, enforce_input_constraints = False, params = [1.0, 0.8, 0.8], plot_x_lim = (-5,5), plot_y_lim = (-5,5), offline = False, offline_iterations = 20, alpha1 = 0.8, alpha2 = 0.8 ):

    t = 0
    
    plt.ion()
    fig = plt.figure()
    ax = plt.axes( xlim = plot_x_lim, ylim = plot_y_lim )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect(1)
    
    robot = Bicycle_2d( X_init, dt_inner, ax, alpha1 = alpha1 )
    global obs1X, obs2X
    obs1 = circle2D(obs1X[0], obs1X[1], d_obs,ax,0)
    obs2 = circle2D(obs2X[0], obs2X[1], d_obs,ax,1)#1.5, 1.9
    params_copy = np.copy( np.asarray(params) )
    
    i = 0
    step_rewards = []
    step_params = np.asarray(params).reshape(-1,1)
    
    initialize_tensors( robot, obs1, obs2, params )
    
    nominal_failed = False
    offline_done = False
    global H    
    metadata = dict(title='Movie Adapt 0', artist='Matplotlib',comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    with writer.saving(fig, movie_name, 100): 

        while (t < tf):
            # print(f"t:{t}")

            i = i + 1
            
            # if (H>2):
            #     H = H - 1

            if ((i % outer_loop != 0) and (not offline)) or ( offline and offline_done ): # compute control input and move the robot
                
                robot.X_torch = torch.tensor(robot.X, dtype=torch.float)
                # get controller
                # control_ref = torch.tensor([0,0], dtype=torch.float).reshape(-1,1)
                control_ref = traced_bicycle2D_nominal_input_jit( robot.X_torch, robot.goal_torch )
                A, b = traced_bicycle2D_qp_constraints_jit( robot.X_torch, robot.goal_torch, obs1.X_torch, obs2.X_torch, torch.tensor(params[0], dtype=torch.float), torch.tensor(params[1], dtype=torch.float), torch.tensor(params[2], dtype=torch.float) )
                # print(f"A:{A}, b:{b}, ref:{control_ref}")
                control, deltas = cbf_controller_layer( control_ref, A, b, solver_args=solver_args )
                if np.any( deltas[1:].detach().numpy() > 0.3 ):
                        nominal_failed = True
                        print(f"Error, ACTUAL controller failed, CHECK algorithm: control:{control.T}, delta:{deltas.T}")
                # print(f"t:{t}, actual: {control.T}") 
                # print(f"control: {control.T}")
                robot.step( control.detach().numpy() )
                robot.render_plot()
                
                step_rewards.append( traced_bicycle_compute_reward_jit( torch.tensor(robot.X_torch, dtype=torch.float), robot.goal_torch, control ).item() )
                step_params = np.append( step_params, np.asarray(params).reshape(-1,1), axis=1 )
                
                
                # Nominal robot
                if not nominal_failed:
                    robot.X_nominal_torch = torch.tensor(robot.X_nominal, dtype=torch.float)
                    # control_ref = torch.tensor([0,0], dtype=torch.float).reshape(-1,1)
                    control_ref = bicycle2D_nominal_input_jit( robot.X_nominal_torch, robot.goal_torch )
                    A, b = traced_bicycle2D_qp_constraints_jit( robot.X_nominal_torch, robot.goal_torch, obs1.X_torch, obs2.X_torch, torch.tensor(params_copy[0], dtype=torch.float), torch.tensor(params_copy[1], dtype=torch.float), torch.tensor(params_copy[2], dtype=torch.float) )
                    control, deltas = cbf_controller_layer( control_ref, A, b )  
                    if np.any( deltas[1:].detach().numpy() > 0.3 ):
                        nominal_failed = True
                        print(f"Error, Nominal controller failed: control:{control.T}, delta:{deltas.T}")
                    # print(f"nominal: {control.T}")          
                    robot.step_nominal( control.detach().numpy() )
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                # writer.grab_frame()
            
                t = t + dt_inner
                
            else: # update parameters with predictive framework
                
                initialize_tensors(robot, obs1, obs2, params )
                
                if (not offline): 
                
                    for k in range(num_gd_iterations):
                        # print(f"k:{k}")    
                        success = False                    
                        while not success:       
                            robot.params = torch.tensor( params, dtype=torch.float, requires_grad=True )     
                            reward, improve_constraints, maintain_constraints, success = compute_reward(robot, obs1, obs2, robot.params, torch.tensor(dt_outer, dtype=torch.float))
                            grads = constrained_update( reward, maintain_constraints, improve_constraints, robot.params )
                            
                            grads = np.clip( grads, -2.0, 2.0 )
                            
                            params[0] = np.clip( params[0] + lr_alpha * grads[0], 0.0, None ).item()
                            params[1] = np.clip( params[1] + lr_alpha * grads[1], 0.0, None ).item()
                            params[2] = np.clip( params[2] + lr_alpha * grads[2], 0.0, None ).item()
                            # print(f"grads: {grads.T}, params: {params}")
                            print(f"params: {params}")
                        print(f"Success!")
                else:
                    
                    for k in range( offline_iterations ):        
                        # print(f"k:{k}")            
                        success = False                    
                        while not success:         
                            robot.params = torch.tensor( params, dtype=torch.float, requires_grad=True )   
                            reward, improve_constraints, maintain_constraints, success = compute_reward(robot, obs1, obs2, robot.params, torch.tensor(dt_outer, dtype=torch.float))
                            grads = constrained_update( reward, maintain_constraints, improve_constraints, robot.params )
                            
                            grads = np.clip( grads, -2.0, 2.0 )
                            
                            params[0] = np.clip( params[0] + lr_alpha * grads[0], 0.0, None ).item()
                            params[1] = np.clip( params[1] + lr_alpha * grads[1], 0.0, None ).item()
                            params[2] = np.clip( params[2] + lr_alpha * grads[2], 0.0, None ).item()
                            print(f"params: {params}")
                        print(f"Success!")
                            
                    offline_done = True
                    
                
    return fig, ax, robot, step_rewards, step_params

            
            
# Run simulations
fig1, ax1, robot1, rewards1, params1 = simulate_scenario( movie_name = 'bicycle_2d/figures/cs4_case1_rc.mp4', adapt=True, enforce_input_constraints=True, params = [1.0, 3.0, 10.0], plot_x_lim = plot_x_lim, plot_y_lim = plot_y_lim, offline = False, offline_iterations=20 )            
# fig1, ax1, robot1, rewards1, params1 = simulate_scenario( movie_name = 'RateTunableCBFs/bicycle_2d/figures/cs4_case1_rc.mp4', adapt=True, enforce_input_constraints=True, params = [1.0, 2.0, 3.0, 10.0], plot_x_lim = plot_x_lim, plot_y_lim = plot_y_lim, offline = False, offline_iterations=20 )            
# fig2, ax2, robot2, rewards2, params2 = simulate_scenario( movie_name = 'bicycle_2d/figures/cs4_case2_rc.mp4', adapt=True, enforce_input_constraints=True, params = [0.5, 0.5, 0.5], plot_x_lim = plot_x_lim, plot_y_lim = plot_y_lim, offline = False, offline_iterations=20 )            
# fig3, ax3, robot3, rewards3, params3 = simulate_scenario( movie_name = 'bicycle_2d/figures/cs4_case1_offline.mp4', adapt=True, enforce_input_constraints=True, params = [1.0, 3.0, 3.0], plot_x_lim = plot_x_lim, plot_y_lim = plot_y_lim, offline = True, offline_iterations=20 )            
# fig4, ax4, robot4, rewards4, params4 = simulate_scenario( movie_name = 'bicycle_2d/figures/cs4_case2_offline.mp4', adapt=True, enforce_input_constraints=True, params = [0.5, 0.5, 0.5], plot_x_lim = plot_x_lim, plot_y_lim = plot_y_lim, offline = True, offline_iterations=20 )

# plt.ioff()

# with open('bicycle_2d/mpc_case1.npy', 'rb') as f:
#     Xs = np.load(f)
    
# fig, ax = plt.subplots(1,1)
# ax.set_xlim( plot_x_lim )
# ax.set_ylim( plot_y_lim )

# # Plot obstacles
# circ = plt.Circle((obs1X[0],obs1X[1]),d_obs, linewidth = 1, edgecolor='k',facecolor='k')
# ax.add_patch(circ)
# circ2 = plt.Circle((obs2X[0],obs2X[1]),d_obs, linewidth = 1, edgecolor='k',facecolor='k')
# ax.add_patch(circ2)

# # Plot MPC solution
# ax.plot(Xs[0,1:], Xs[1,1:],'r', label='MPC')

# # Plot new solution

# # Case 1
# ax.plot(robot1.Xs_nominal[0,:], robot1.Xs_nominal[1,:], 'g', label='Nominal Case 1')
# ax.plot(robot1.Xs[0,:], robot1.Xs[1,:], 'g.', label='RC Case 1')
# ax.plot(robot3.Xs[0,:], robot3.Xs[1,:], 'g-.', label='Fixed tuned Case 1')

# # Case 2
# ax.plot(robot2.Xs_nominal[0,:], robot2.Xs_nominal[1,:], 'b', label='Nominal Case 2')
# ax.plot(robot2.Xs[0,:], robot2.Xs[1,:], 'b.', label='RC Case 2')
# ax.plot(robot4.Xs[0,:], robot4.Xs[1,:], 'b-.', label='Fixed tuned Case 2')



# # Show plot
# ax.legend()
# fig.savefig("bicycle_2d/figures/cs4.png")
# plt.show()


