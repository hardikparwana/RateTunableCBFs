import numpy as np
import torch

# @torch.jit.script    
def unicycle_f_torch_jit(x):
    return torch.tensor([0.0,0.0,0.0],dtype=torch.float).reshape(-1,1)

# @torch.jit.script
def unicycle_g_torch_jit(x):
    # return torch.tensor([ [torch.cos(x[2,0]),0.0],[torch.sin(x[2,0]),0.0],[0,1] ])
    g1 = torch.cat( (torch.cos(x[2,0]).reshape(-1,1),torch.tensor([[0]]) ), dim=1 )
    # g2 = torch.cat( ( torch.tensor([[0]]), torch.sin(x[2,0]).reshape(-1,1) ), dim=1 )
    g2 = torch.cat( ( torch.sin(x[2,0]).reshape(-1,1), torch.tensor([[0]]) ), dim=1 )
    g3 = torch.tensor([[0,1]],dtype=torch.float)
    gx = torch.cat((g1,g2,g3))
    return gx

def update_unicycle_state_jit( state, control, dt ):
    f = unicycle_f_torch_jit(state)
    g = unicycle_g_torch_jit(state)
    return state + ( f + g @ control ) * dt
traced_update_unicycle_state_jit = torch.jit.trace( update_unicycle_state_jit, ( torch.ones(3,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.tensor(0.5, dtype=torch.float) ) )

# @torch.jit.script
def unicycle2D_lyapunov_jit(X, G, min_D = torch.tensor(0.3, dtype=torch.float), max_D = torch.tensor(2.0, dtype=torch.float)):   
    V = torch.square ( torch.norm( X[0:2] - G[0:2] ) )
    factor = 2 * (  X[0:2] - G[0:2] ).reshape(1,-1) 
    dV_dxi = torch.cat( (factor, torch.tensor([[0]])), dim  = 1 )
    dV_dxj = -factor
    
    return V, dV_dxi, dV_dxj

# @torch.jit.script
def sigma_torch(s):
    k1 = torch.tensor(3.0, dtype=torch.float)
    k2 = torch.tensor(0.5, dtype=torch.float)
    return k2 * torch.div( (torch.exp(k1-s)-1) , (torch.exp(k1-s)+1) )

# @torch.jit.script
def sigma_der_torch(s):
    k1 = torch.tensor(3.0, dtype=torch.float) # 4.0
    k2 = torch.tensor(0.5, dtype=torch.float) # 1.0
    return - k2 * torch.div ( torch.exp(k1-s),( 1+torch.exp( k1-s ) ) ) @ ( 1 - sigma_torch(s)/k2 )

# @torch.jit.script
def unicycle_SI2D_barrier_jit(X, targetX):#, min_D = torch.tensor(0.3, dtype=torch.float)): # target is unicycle
    
    min_D = torch.tensor(0.3, dtype=torch.float)
    beta = 1.01
    h = torch.square( torch.norm(X[0:2] - targetX[0:2])  ) - beta*min_D**2
    h1 = h
    
    theta = X[2,0]
    s = (X[0:2] - targetX[0:2]).T @ torch.cat( ( torch.cos(X[2,0]).reshape(-1,1), torch.sin(X[2,0]).reshape(-1,1) ) )
    h_final = h - sigma_torch(s)
    # print(f"h1:{h1}, h:{h_final}")
    if h_final<0:
           h_final = 0 * h_final
    
    der_sigma = sigma_der_torch(s)
    
    dh_dxi = torch.cat( ( 2*( X[0:2] - targetX[0:2] ).T - der_sigma @ torch.cat( (torch.cos(X[2,0]).reshape(-1,1), torch.sin(X[2,0]).reshape(-1,1)),1 ) ,  - der_sigma * ( -torch.sin(X[2,0]).reshape(-1,1) @ ( X[0,0]-targetX[0,0] ).reshape(-1,1) + torch.cos(X[2,0]).reshape(-1,1) @ ( X[1,0] - targetX[1,0] ).reshape(-1,1) ) ), 1)
    dh_dxj = -2*( X[0:2] - targetX[0:2] ).T + der_sigma @ torch.cat( (torch.cos(X[2,0]).reshape(-1,1), torch.sin(X[2,0]).reshape(-1,1)),1 )
    # print(f"h1:{h1}, h:{h_final}, dh_dx:{ dh_dxi }")
    return h_final, dh_dxi, dh_dxj

# @torch.jit.script
def unicycle_compute_reward_jit(X,targetX, control):
    # return 100 * torch.square( torch.norm( X[0:2,0] - targetX[0:2,0]  ) ) #- torch.tensor((min_D+max_D)/2) ) - 2 * h3
    return 1000 * torch.square( torch.norm( X[0:2,0] - targetX[0:2,0]  ) ) + torch.square( torch.norm(control) )
traced_unicycle_compute_reward_jit = torch.jit.trace( unicycle_compute_reward_jit, ( torch.ones(3,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float) ) )
    
    
def unicycle_nominal_input_jit(X, targetX):#, d_min = torch.tensor(0.3, dtype=torch.float)):
    d_min = torch.tensor(0.3, dtype=torch.float)
    k_omega = 2.0 #0.5#2.5
    k_v = 3.0 #2.0 #0.5
    distance = torch.norm( targetX - X[0:2] ) - d_min
    theta_d = torch.arctan2(targetX[1,0]-X[1,0],targetX[0,0]-X[0,0])
    error_theta = torch.atan2( torch.sin(theta_d-X[2,0]), torch.cos(theta_d-X[2,0]) )

    omega = k_omega*error_theta   
    v = k_v*( distance )*torch.cos( error_theta )
    return torch.cat( (v.reshape(-1,1),omega.reshape(-1,1)), dim=0 )
traced_unicycle_nominal_input_jit = torch.jit.trace( unicycle_nominal_input_jit, ( torch.ones(3,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float)  ) ) #, torch.tensor(0.3, dtype=torch.float)

def unicycle2D_qp_constraints_jit(state, goal, obs1, obs2, param0, param1, param2):
    
    V, dV_dx, _ = unicycle2D_lyapunov_jit( state, goal )
    h1, dh1_dx, _ = unicycle_SI2D_barrier_jit( state, obs1  )
    h2, dh2_dx, _ = unicycle_SI2D_barrier_jit( state, obs2  )
    f = unicycle_f_torch_jit(state)
    g = unicycle_g_torch_jit(state)
    A0 = -dV_dx @ g; b0 = -dV_dx @ f - param0 * V
    A1 = dh1_dx @ g; b1 = dh1_dx @ f + param1 * h1
    A2 = dh2_dx @ g; b2 = dh2_dx @ f + param2 * h2        
    A = torch.cat( (A0, A1, A2), dim=0 )
    b = torch.cat( (b0, b1, b2), dim=0 )
    return A, b
traced_unicycle2D_qp_constraints_jit = torch.jit.trace( unicycle2D_qp_constraints_jit, ( torch.ones(3,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.tensor(0.5, dtype=torch.float), torch.tensor(0.5, dtype=torch.float), torch.tensor(0.5, dtype=torch.float) ) )

# @torch.jit.script
def unicycle_fov_compute_reward_jit(X,targetX):
    
    max_D = torch.tensor(2.0, dtype=torch.float) #2.0
    min_D = torch.tensor(0.3, dtype=torch.float) #0.3
    FoV_angle = torch.tensor(3.14157/3, dtype=torch.float) #3.13/3    

    p = targetX[0:2] - X[0:2]
    dir_vector = torch.cat( ( torch.cos(X[2,0]).reshape(-1,1), torch.sin(X[2,0]).reshape(-1,1) ) )
    bearing_angle  = torch.matmul(dir_vector.T , p )/ torch.norm(p)
    h3 = (bearing_angle - torch.cos(FoV_angle/2))/(1.0-torch.cos(FoV_angle/2))
    
    return torch.square( torch.norm( X[0:2,0] - targetX[0:2,0]  ) - torch.tensor((min_D+max_D)/2) ) - 2 * h3
    
traced_unicycle_fov_compute_reward_jit = torch.jit.trace( unicycle_fov_compute_reward_jit, ( torch.ones(3,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float) ) )
    

# @torch.jit.script
def unicycle_si_fov_barrier_jit(X, targetX):
    
    max_D = torch.tensor(2.0, dtype=torch.float)#2.0
    min_D = torch.tensor(0.3, dtype=torch.float)#0.3
    FoV_angle = torch.tensor(3.14157/3, dtype=torch.float)#3.14157/3
    
    # Max distance
    h1 = max_D**2 - torch.square( torch.norm( X[0:2] - targetX[0:2] ) )
    dh1_dxi = torch.cat( ( -2*( X[0:2] - targetX[0:2] ), torch.tensor([[0.0]]) ), 0).T
    dh1_dxj =  2*( X[0:2] - targetX[0:2] ).T
    
    # Min distance
    h2 = torch.square(torch.norm( X[0:2] - targetX[0:2] )) - min_D**2
    dh2_dxi = torch.cat( ( 2*( X[0:2] - targetX[0:2] ), torch.tensor([[0.0]]) ), 0).T
    dh2_dxj = - 2*( X[0:2] - targetX[0:2]).T

    # Max angle
    p = targetX[0:2] - X[0:2]

    dir_vector = torch.cat( ( torch.cos(X[2,0]).reshape(-1,1), torch.sin(X[2,0]).reshape(-1,1) ) )
    bearing_angle  = torch.matmul(dir_vector.T , p )/ torch.norm(p)
    h3 = (bearing_angle - torch.cos(FoV_angle/2))/(1.0-torch.cos(FoV_angle/2))

    norm_p = torch.norm(p)
    dh3_dx = dir_vector.T / norm_p - ( dir_vector.T @ p)  * p.T / torch.pow(norm_p,3)    
    dh3_dTheta = ( -torch.sin(X[2]) * p[0] + torch.cos(X[2]) * p[1] ).reshape(1,-1)  /torch.norm(p)
    dh3_dxi = torch.cat(  ( -dh3_dx , dh3_dTheta), 1  ) /(1.0-torch.cos(FoV_angle/2))
    dh3_dxj = dh3_dx /(1.0-torch.cos(FoV_angle/2))
        
    return h1, dh1_dxi, dh1_dxj, h2, dh2_dxi, dh2_dxj, h3, dh3_dxi, dh3_dxj

def unicycle2D_LF_qp_constraints_jit(stateF, stateL, stateLdot, param0, param1, param2, param3):
    
    V, dV_dxF, dV_dxL = unicycle2D_lyapunov_jit( stateF, stateL)
    h1, dh1_dxF, dh1_dxL, h2, dh2_dxF, dh2_dxL, h3, dh3_dxF, dh3_dxL = unicycle_si_fov_barrier_jit( stateF, stateL )
    Ff = unicycle_f_torch_jit(stateF)
    Fg = unicycle_g_torch_jit(stateF)
    A0 = -dV_dxF @ Fg; b0 = -dV_dxF @ Ff - dV_dxL @ stateLdot - param0 * V
    A1 = dh1_dxF @ Fg; b1 = dh1_dxF @ Ff + dh1_dxL @ stateLdot + param1 * h1
    A2 = dh2_dxF @ Fg; b2 = dh2_dxF @ Ff + dh2_dxL @ stateLdot + param2 * h2
    A3 = dh3_dxF @ Fg; b3 = dh3_dxF @ Ff + dh3_dxL @ stateLdot + param3 * h3
    A = torch.cat( (A0, A1, A2, A3), dim=0 )
    b = torch.cat( (b0, b1, b2, b3), dim=0 )
    return A, b
traced_unicycle2D_LF_qp_constraints_jit = torch.jit.trace( unicycle2D_LF_qp_constraints_jit, ( torch.ones(3,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.tensor(0.5, dtype=torch.float), torch.tensor(0.5, dtype=torch.float), torch.tensor(0.5, dtype=torch.float), torch.tensor(0.5, dtype=torch.float) ) )
