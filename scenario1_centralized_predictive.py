# better do JIT for speed
# still need multi-agent approach for uncooperative and adversarial agents
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
torch.autograd.set_detect_anomaly(True)

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning) 



def constrained_update( objective, maintain_constraints, improve_constraints, params ) :
    
    
    num_params = 4
    d = cp.Variable((num_params,1))
    
    # Get Performance optimal direction
    try:
        objective.sum().backward(retain_graph = True) 
        k_grad = getGrad(params[0], l_bound = -20.0, u_bound = 20.0 )
        alpha_grad = getGrad(params[1], l_bound = -20.0, u_bound = 20.0 )
        objective_grad = np.append( k_grad.reshape(1,-1), alpha_grad.reshape(1,-1), axis = 1 )
    except:
        objective_grad = np.array([[0,0,0,0]])
    
    # Get constraint improve direction # assume one at a time
    improve_constraint_direction = np.array([0,0,0,0]).reshape(1,-1)
    for i, constraint in enumerate( improve_constraints):
        constraint.sum().backward(retain_graph=True)
        k_grad = getGrad(params[0], l_bound = -20.0, u_bound = 20.0 )
        alpha_grad = getGrad(params[1], l_bound = -20.0, u_bound = 20.0 )
        improve_constraint_direction = improve_constraint_direction +  np.append( k_grad.reshape(1,-1), alpha_grad.reshape(1,-1), axis = 1 )
    
    # Get allowed directions
    N = len(maintain_constraints)
    if N>0:
        d_maintain = np.zeros((N,num_params))#cp.Variable( (N, num_params) )
        constraints = []
        for i, constraint in enumerate(maintain_constraints):
            constraint.sum().backward(retain_graph=True)
            k_grad = getGrad(params[0], l_bound = -20.0, u_bound = 20.0 )
            alpha_grad = getGrad(params[1], l_bound = -20.0, u_bound = 20.0 )
            d_maintain[i,:] = np.append( k_grad.reshape(1,-1), alpha_grad.reshape(1,-1), axis = 1 )[0]
            
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
        
        # print("update direction: ", d.value.T)
        
        return d.value
    
    else:
        if len( improve_constraints ) > 0:
            obj = cp.Maximize( improve_constraint_direction @ d )
            # print("update direction: ", -improve_constraint_direction.reshape(-1,1).T)
            return -improve_constraint_direction.reshape(-1,1)
        else:
            return -objective_grad.reshape(-1,1)
    
def get_future_reward()