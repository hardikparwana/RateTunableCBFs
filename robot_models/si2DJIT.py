import numpy as np
import torch

# @torch.jit.script    
def si_f_torch_jit(x):
    return torch.tensor([0.0,0.0],dtype=torch.float).reshape(-1,1)

# @torch.jit.script
def si_g_torch_jit(x):
    # return torch.tensor([ [torch.cos(x[2,0]),0.0],[torch.sin(x[2,0]),0.0],[0,1] ])
    g1 = torch.tensor([[1,0]],dtype=torch.float)
    g2 = torch.tensor([[0,1]],dtype=torch.float)
    gx = torch.cat((g1,g2))
    return gx

def update_si_state_jit( state, control, dt ):
    f = si_f_torch_jit(state)
    g = si_g_torch_jit(state)
    return state + ( f + g @ control ) * dt
traced_update_si_state_jit = torch.jit.trace( update_si_state_jit, ( torch.ones(2,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.tensor(0.5, dtype=torch.float) ) )

def SI_leader_motion(t):
    u = torch.tensor([[2]], dtype=torch.float)
    v = 3*torch.cos(5*t).reshape(-1,1)
    return torch.cat( (u,v), dim=0 )

# @torch.jit.script
def si2D_lyapunov_jit(X, G, min_D = torch.tensor(0.3, dtype=torch.float), max_D = torch.tensor(2.0, dtype=torch.float)):   
    V = torch.square ( torch.norm( X[0:2] - G[0:2] ) )
    dV_dxi = 2 * (  X[0:2] - G[0:2] ).reshape(1,-1) 
    dV_dxj = - dV_dxi
    
    return V, dV_dxi, dV_dxj
# @torch.jit.script
def si_SI2D_barrier_jit(X, targetX):#, min_D = torch.tensor(0.3, dtype=torch.float)): # target is si
    min_D = torch.tensor(0.3, dtype=torch.float)
    h = torch.square( torch.norm(X[0:2] - targetX[0:2])  ) - min_D**2
    dh_dxi = 2*( X[0:2] - targetX[0:2] ).T
    dh_dxj = - dh_dxi
    return h, dh_dxi, dh_dxj

# @torch.jit.script
def si_compute_reward_jit(X,targetX, control):
    # return 100 * torch.square( torch.norm( X[0:2,0] - targetX[0:2,0]  ) ) #- torch.tensor((min_D+max_D)/2) ) - 2 * h3
    return 100 * torch.square( torch.norm( X[0:2,0] - targetX[0:2,0]  ) ) + torch.square( torch.norm(control) )
traced_si_compute_reward_jit = torch.jit.trace( si_compute_reward_jit, ( torch.ones(2,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float) ) )
    
def si2D_qp_constraints_jit(state, goal, obs1, obs2, param0, param1, param2):
    
    V, dV_dx, _ = si2D_lyapunov_jit( state, goal )
    h1, dh1_dx, _ = si_SI2D_barrier_jit( state, obs1  )
    h2, dh2_dx, _ = si_SI2D_barrier_jit( state, obs2  )
    f = si_f_torch_jit(state)
    g = si_g_torch_jit(state)
    A0 = -dV_dx @ g; b0 = -dV_dx @ f - param0 * V
    A1 = dh1_dx @ g; b1 = dh1_dx @ f + param1 * torch.pow(h1,3)
    A2 = dh2_dx @ g; b2 = dh2_dx @ f + param2 * torch.pow(h2,3)        
    A = torch.cat( (A0, A1, A2), dim=0 )
    b = torch.cat( (b0, b1, b2), dim=0 )
    return A, b
traced_si2D_qp_constraints_jit = torch.jit.trace( si2D_qp_constraints_jit, ( torch.ones(2,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.tensor(0.5, dtype=torch.float), torch.tensor(0.5, dtype=torch.float), torch.tensor(0.5, dtype=torch.float) ) )
