import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import matplotlib.pyplot as plt
import time
from robot_models.obstacles import circle2D
from utils.utils import *
from robot_models.si2DJIT import *
from robot_models.si_mpc import SI

dt_inner = 0.05
N = 100
tf =  int( 100 * dt_inner ) #20
outer_loop = 10
num_gd_iterations = 5
dt_outer = 0.1
H = 100
lr_alpha = 0.1#0.05
plot_x_lim = (-0.5,3.5)  
plot_y_lim = (-0.5,3.5) 

# starting point
# X_init = np.array([-0.5,-0.5,np.pi/2])
X_init = np.array([-0.5,-0.5])
d_obs = 0.3
goalX = np.array([2.0,2.0])

# input bounds
u1_max = 2
u2_max = 5


##  Define Controller ################

# si
num_constraints = 3
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
    
    for i in range(H):

        # make  control matrices
        control_ref = torch.tensor([0,0], dtype=torch.float).reshape(-1,1)
        A, b = traced_si2D_qp_constraints_jit( states[i], robot.goal_torch, obs1.X_torch, obs2.X_torch, params[0], params[1], params[2] )
        control, deltas = cbf_controller_layer( control_ref, A, b )
        
        # Check for constraints that need to be maintained or kept
        # if np.any( deltas[1:].detach().numpy() > 0.01 ):
        #     # if improve_constraints == []:
        #     #     improve_constraints = 
        #     improve_constraints = torch.cat( improve_constraints )
        #     return reward, improve_constraints, maintain_constraints, False
        # else:
        #     maintain_constraints = torch.cat( (maintain_constraints, A @ control), dim=0 )
                   
        # Get next state
        next_state = update_si_state_jit( states[i], control, dt_outer )
                
        # Save next state and compute reward
        states.append( next_state )
        reward = reward + traced_si_compute_reward_jit( states[i+1], robot.goal_torch, control )       
        
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
    
    robot = SI( X_init, dt_inner, ax )
    obs1 = circle2D(0.7,0.7,d_obs,ax,0)
    obs2 = circle2D(1.5,1.9,d_obs,ax,1)#1.5, 1.9
    params_copy = np.copy( np.asarray(params) )
    
    i = 0
    step_rewards = []
    step_params = np.asarray(params).reshape(-1,1)
    
    initialize_tensors( robot, obs1, obs2, params )
    
    offline_done = False

    while (t < tf):

        i = i + 1

        if ((i % outer_loop != 0) and (not offline)) or ( offline and offline_done ): # compute control input and move the robot
            
            robot.X_torch = torch.tensor(robot.X, dtype=torch.float)
        
            # get controller
            control_ref = torch.tensor([0,0], dtype=torch.float).reshape(-1,1)
            A, b = si2D_qp_constraints_jit( robot.X_torch, robot.goal_torch, obs1.X_torch, obs2.X_torch, torch.tensor(params[0], dtype=torch.float), torch.tensor(params[1], dtype=torch.float), torch.tensor(params[2], dtype=torch.float) )
            control, deltas = cbf_controller_layer( control_ref, A, b )
            
            # print(f"control: {control.T}")
            robot.step( control.detach().numpy() )
            robot.render_plot()
            
            step_rewards.append( si_compute_reward_jit( torch.tensor(robot.X_torch, dtype=torch.float), robot.goal_torch, control ).item() )
            step_params = np.append( step_params, np.asarray(params).reshape(-1,1), axis=1 )
            
            
            # Nominal robot
            robot.X_nominal_torch = torch.tensor(robot.X_nominal, dtype=torch.float)
            control_ref = torch.tensor([0,0], dtype=torch.float).reshape(-1,1)
            A, b = si2D_qp_constraints_jit( robot.X_nominal_torch, robot.goal_torch, obs1.X_torch, obs2.X_torch, torch.tensor(params_copy[0], dtype=torch.float), torch.tensor(params_copy[1], dtype=torch.float), torch.tensor(params_copy[2], dtype=torch.float) )
            control, deltas = cbf_controller_layer( control_ref, A, b )            
            robot.step_nominal( control.detach().numpy() )
            
            fig.canvas.draw()
            fig.canvas.flush_events()
        
            t = t + dt_inner
            
        else: # update parameters with predictive framework
            
            initialize_tensors(robot, obs1, obs2, params )
            
            if (not offline): 
            
                for k in range(num_gd_iterations):
                    success = False                    
                    while not success:            
                        reward, improve_constraints, maintain_constraints, success = compute_reward(robot, obs1, obs2, robot.params, torch.tensor(dt_outer, dtype=torch.float))
                        grads = constrained_update( reward, maintain_constraints, improve_constraints, robot.params )
                        
                        grads = np.clip( grads, -2.0, 2.0 )
                        
                        params[0] = np.clip( params[0] + lr_alpha * grads[0], 0.0, None )
                        params[1] = np.clip( params[1] + lr_alpha * grads[1], 0.0, None )
                        params[2] = np.clip( params[2] + lr_alpha * grads[2], 0.0, None )
                        # print(f"grads: {grads.T}, params: {params}")
            else:
                
                for k in range( offline_iterations ):        
                    print(f"k:{k}")            
                    success = False                    
                    while not success:            
                        reward, improve_constraints, maintain_constraints, success = compute_reward(robot, obs1, obs2, robot.params, torch.tensor(dt_outer, dtype=torch.float))
                        grads = constrained_update( reward, maintain_constraints, improve_constraints, robot.params )
                        
                        grads = np.clip( grads, -2.0, 2.0 )
                        
                        params[0] = np.clip( params[0] + lr_alpha * grads[0], 0.0, None )
                        params[1] = np.clip( params[1] + lr_alpha * grads[1], 0.0, None )
                        params[2] = np.clip( params[2] + lr_alpha * grads[2], 0.0, None )
                        # print(f"grads: {grads.T}, params: {params}")
                        
                offline_done = True
                    
                
    return fig, ax, robot, step_rewards, step_params

            
            
# Run simulations
fig1, ax1, robot1, rewards1, params1 = simulate_scenario( movie_name = 'test.mp4', adapt=True, enforce_input_constraints=True, params = [1.0, 3.0, 3.0], plot_x_lim = plot_x_lim, plot_y_lim = plot_y_lim, offline = False, offline_iterations=20 )            