import numpy as np
import torch


# @torch.jit.script    
def bicycle_f_torch_jit(x):
    zero = torch.tensor([[0]], dtype=torch.float)
    f = torch.cat( ( (x[3,0]*torch.cos(x[2,0])).reshape(-1,1),
                     (x[3,0]*torch.sin(x[2,0])).reshape(-1,1),
                     zero,
                     zero    
                    )  , dim=0)
    return f 

# @torch.jit.script
def bicycle_g_torch_jit(x):
    # return torch.tensor([ [torch.cos(x[2,0]),0.0],[torch.sin(x[2,0]),0.0],[0,1] ])
    g1 = torch.tensor([[0,0]],dtype=torch.float)
    g2 = torch.tensor([[0,0]],dtype=torch.float)
    g3 = torch.tensor([[0,1]],dtype=torch.float)
    g4 = torch.tensor([[1,0]],dtype=torch.float)
    gx = torch.cat((g1,g2,g3,g4), dim=0)
    return gx

def update_bicycle_state_jit( state, control, dt ):
    f = bicycle_f_torch_jit(state)
    g = bicycle_g_torch_jit(state)
    next_state = state + ( f + g @ control ) * dt
    next_state_copy = next_state.clone()
    next_state[2,0] = torch.atan2(torch.sin(next_state_copy[2,0]), torch.cos(next_state_copy[2,0]))
    # print(f"state:{state.T}, next_state:{next_state.T}, f:{f.T}")
    return next_state
traced_update_si_state_jit = torch.jit.trace( update_bicycle_state_jit, ( torch.ones(4,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.tensor(0.5, dtype=torch.float) ) )

def bicycle_barrier_jit(X, targetX, alpha1):#, min_D = torch.tensor(0.3, dtype=torch.float)): # target is si
    
    d_min = torch.tensor(0.3, dtype=torch.float)
    zero = torch.tensor([[0]], dtype=torch.float)
    one = torch.tensor([[1]], dtype=torch.float)
    f = torch.cat((X[3,0]*torch.cos(X[2,0]).reshape(-1,1),
                         X[3,0]*torch.sin(X[2,0]).reshape(-1,1),
                         zero,
                       zero), dim=0)
        
    # 4 x 4
    Df_dx = torch.cat(( 
                        torch.cat(( zero, zero, -X[3,0]*torch.sin(X[2,0]).reshape(-1,1), torch.cos(X[2,0]).reshape(-1,1) ), dim=1 ),
                        torch.cat(( zero, zero,  X[3,0]*torch.cos(X[2,0]).reshape(-1,1), torch.sin(X[2,0]).reshape(-1,1) ), dim=1 ),
                        torch.tensor([[0,0,0,0]], dtype=torch.float),
                        torch.tensor([[0,0,0,0]], dtype=torch.float)
                ), dim=0)
    
    h = torch.norm( X[0:2] - targetX[0:2] )**2 - d_min**2     
    # print(f"h:{h}")      
    dh_dx = torch.cat( (2*( X[0:2] - targetX[0:2] ).T, zero, zero), dim=1 )
                    
    # single derivative
    x_dot = f[0,0] # u cos(theta)
    y_dot = f[1,0] # u sin(theta)
    dx_dot_dx = Df_dx[0,:].reshape(1,-1)
    dy_dot_dx = Df_dx[1,:].reshape(1,-1)
    
    h_dot = 2*(X[0,0]-targetX[0,0])*x_dot + 2*(X[1,0]-targetX[1,0])*y_dot
    dh_dot_dx = 2*(X[0,0]-targetX[0,0])*dx_dot_dx + 2*x_dot*torch.tensor([[1,0,0,0]], dtype=torch.float) + 2*(X[2,0]-targetX[1,0])*dy_dot_dx + 2*y_dot*torch.tensor([[0,1,0,0]], dtype=torch.float)
    
    h2 = h_dot + alpha1 * h
    dh2_dx = dh_dot_dx + alpha1 * dh_dx
    
    return h.reshape(-1,1), h2.reshape(-1,1), dh2_dx

def bicycle2D_nominal_input_jit(X,target, k_v=torch.tensor(3.0, dtype=torch.float), k_omega=torch.tensor(5.0, dtype=torch.float),):
    
        x = X[0,0]
        y = X[1,0]
        theta = X[2,0]
        v = X[3,0]
        d_min = torch.tensor(0.01, dtype=torch.float)
        
        # k_omega = torch.tensor(5.0, dtype=torch.float) #0.5#2.5
        # k_v = torch.tensor(3.0, dtype=torch.float) #2.0 #0.5
        
        theta_d = torch.atan2(target[1,0]-X[1,0],target[0,0]-X[0,0])
        error_angle = theta_d - X[2,0]
        error_theta = torch.atan2( torch.sin(error_angle), torch.cos(error_angle) )
        omega = k_omega*error_theta  
        
        zero = torch.tensor([[0]], dtype=torch.float)
        distance_temp = torch.norm( X[0:2,0]-target[0:2,0] ).reshape(-1,1) - d_min
        distance = torch.max( torch.cat( (distance_temp, zero) ) )
        speed = k_v*distance*torch.cos( error_theta )
        u_r = k_v * ( speed - v )
        # print(f"u_r;{u_r}, omega:{omega}")
        return torch.cat((u_r.reshape(-1,1), omega.reshape(-1,1)), dim=0)
    
traced_bicycle2D_nominal_input_jit = torch.jit.trace( bicycle2D_nominal_input_jit, ( torch.ones(4,1, dtype=torch.float), torch.zeros(2,1, dtype=torch.float), torch.tensor(0.5, dtype=torch.float), torch.tensor(0.5, dtype=torch.float) ) ) 
            
def bicycle_lyapunov_jit(X, G):
    # print("hell1")
    zero = torch.tensor([[0]], dtype=torch.float)
    V = torch.square(torch.norm( X[0:2] - G[0:2] ))
    dV_dx = torch.cat( (2*(X[0:2]-G[0:2]).T, zero , zero) , dim=1       )
    return V, dV_dx
    
# @torch.jit.script
def bicycle_compute_reward_jit(X,targetX, control):
    # return 100 * torch.square( torch.norm( X[0:2,0] - targetX[0:2,0]  ) ) #- torch.tensor((min_D+max_D)/2) ) - 2 * h3
    return 100 * torch.square( torch.norm( X[0:2,0] - targetX[0:2,0]  ) ) + torch.square( torch.norm(control) )
traced_bicycle_compute_reward_jit = torch.jit.trace( bicycle_compute_reward_jit, ( torch.ones(7,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float) ) )
    
def bicycle2D_qp_constraints_jit(state, goal, obs1, obs2, param0, param1_1, param1_2, param2_1, param2_2): #k, alpha1, alpha2, alpha3
    # print("hell")
    V, dV_dx = bicycle_lyapunov_jit( state, goal )
    h1_1, h1, dh1_dx = bicycle_barrier_jit( state, obs1, param1_1  )
    h2_1, h2, dh2_dx = bicycle_barrier_jit( state, obs2, param2_1  )
    f = bicycle_f_torch_jit(state)
    g = bicycle_g_torch_jit(state)
    A0 = -dV_dx @ g; b0 = -dV_dx @ f - param0 * V
    
    A1 = dh1_dx @ g; b1 = dh1_dx @ f + param1_2 * h1
    # A1 = dh1_dx @ g; b1 = dh1_dx @ f + param1_2 * torch.pow(h1,3)
    A1_1 = A1 * 0; b1_1 = h1_1
    
    A2 = dh2_dx @ g; b2 = dh2_dx @ f + param2_2 * h2#torch.pow(h2,3)
    A2_1 = A2 * 0; b2_1 = h2_1
    
    A = torch.cat( (A0, 1*A1_1, 1*A1, 1*A2_1, 1*A2), dim=0 )
    b = torch.cat( (b0, 1*b1_1, 1*b1, 1*b2_1, 1*b2), dim=0 )
    # A = torch.cat( (A0, A1), dim=0 )
    # b = torch.cat( (b0, b1), dim=0 )
    return A, b
traced_bicycle2D_qp_constraints_jit = torch.jit.trace( bicycle2D_qp_constraints_jit, ( torch.ones(4,1, dtype=torch.float), torch.zeros(2,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.tensor(0.5, dtype=torch.float), torch.tensor(0.5, dtype=torch.float), torch.tensor(0.5, dtype=torch.float), torch.tensor(0.5, dtype=torch.float), torch.tensor(0.5, dtype=torch.float) ) )
