import numpy as np
import torch

V_w = 0.0#0.1
theta_w = torch.tensor(np.pi/3, dtype=torch.float)
m11 = torch.tensor(5.5404, dtype=torch.float)
m22 = torch.tensor(9.6572, dtype=torch.float)
m33 = torch.tensor(0.0574, dtype=torch.float)
X_u = torch.tensor(-2.3015, dtype=torch.float)
X_u_u = torch.tensor(-8.2845, dtype=torch.float)
Y_v = torch.tensor(-8.0149, dtype=torch.float)
Y_v_v = torch.tensor(-23.689, dtype=torch.float)
N_r = torch.tensor(-0.0048, dtype=torch.float)
N_r_r = torch.tensor(-0.0089, dtype=torch.float)

# @torch.jit.script    
def uav_f_torch_jit(x):
    
    f = torch.cat( ( (x[3,0]*torch.cos(x[2,0])-x[4,0]*torch.sin(x[2,0])+V_w*torch.cos(theta_w)).reshape(-1,1),
                       (x[3,0]*torch.sin(x[2,0])+x[4,0]*torch.cos(x[2,0])+V_w*torch.sin(theta_w)).reshape(-1,1),
                       (x[5,0]).reshape(-1,1),
                       ((m22*x[4,0]*x[5,0]+X_u*x[3,0]+X_u_u*torch.abs(x[3,0])*x[3,0]+x[6,0])/m11).reshape(-1,1),
                       ((-m11*x[3,0]*x[5,0]+Y_v*x[4,0]+Y_v_v*torch.abs(x[4,0])*x[4,0])/m22).reshape(-1,1),
                       (((m11-m22)*x[3,0]*x[4,0]+N_r*x[5,0]+N_r_r*torch.abs(x[5,0])*x[5,0])/m33).reshape(-1,1),
                       torch.tensor([[0]], dtype=torch.float)        
                    )  , dim=0)
    
    return f 

# @torch.jit.script
def uav_g_torch_jit(x):
    # return torch.tensor([ [torch.cos(x[2,0]),0.0],[torch.sin(x[2,0]),0.0],[0,1] ])
    g1 = torch.tensor([[0,0]],dtype=torch.float)
    g2 = torch.tensor([[0,0]],dtype=torch.float)
    g3 = torch.tensor([[0,0]],dtype=torch.float)
    g4 = torch.tensor([[0,0]],dtype=torch.float)
    g5 = torch.tensor([[0,0]],dtype=torch.float)
    g6 = torch.tensor([[0,1/m33]],dtype=torch.float)
    g7 = torch.tensor([[1,0]],dtype=torch.float)
    gx = torch.cat((g1,g2,g3,g4,g5,g6,g7))
    return gx

def update_uav_state_jit( state, control, dt ):
    f = uav_f_torch_jit(state)
    g = uav_g_torch_jit(state)
    next_state = state + ( f + g @ control ) * dt
    print(f"state:{state.T}, next_state:{next_state.T}, f:{f.T}")
    return next_state
traced_update_si_state_jit = update_uav_state_jit#torch.jit.trace( update_uav_state_jit, ( torch.ones(7,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.tensor(0.5, dtype=torch.float) ) )

# @torch.jit.script
# def uav2D_lyapunov_jit(X, G, min_D = torch.tensor(0.3, dtype=torch.float), max_D = torch.tensor(2.0, dtype=torch.float)):   
#     V = torch.square ( torch.norm( X[0:2] - G[0:2] ) )
#     dV_dxi = 2 * (  X[0:2] - G[0:2] ).reshape(1,-1) 
#     dV_dxj = - dV_dxi
    
#     return V, dV_dxi, dV_dxj
# @torch.jit.script
def uav_SI2D_barrier_jit(X, targetX, alpha1, alpha2):#, min_D = torch.tensor(0.3, dtype=torch.float)): # target is si
    min_D = torch.tensor(0.3, dtype=torch.float)
    h = torch.square( torch.norm(X[0:2] - targetX[0:2])  ) - min_D**2
    dh_dxi = 2*( X[0:2] - targetX[0:2] ).T
    dh_dxj = - dh_dxi
    # return h, dh_dxi, dh_dxj
    
    f = uav_f_torch_jit(X)
    
    zero = torch.tensor([[0]], dtype=torch.float)
    one = torch.tensor([[1]], dtype=torch.float)
    
    xi = X[0,0]
    yi = X[1,0]
    phi = X[2,0]
    ui = X[3,0]
    vi = X[4,0]
    ri = X[5,0]
    
    Df_dx = torch.cat((  torch.cat(( zero, zero, (-X[3,0]*torch.sin(X[2,0])-X[4,0]*torch.cos(X[2,0])).reshape(-1,1), torch.cos(X[2,0]).reshape(-1,1), -torch.sin(X[2,0]).reshape(-1,1), zero, zero), dim=1),
                         torch.cat(( zero, zero, ( X[3,0]*torch.cos(X[2,0])-X[4,0]*torch.sin(X[2,0])).reshape(-1,1), torch.sin(X[2,0]).reshape(-1,1), torch.cos(X[2,0]).reshape(-1,1), zero, zero), dim=1),
                         torch.cat(( zero, zero, zero, zero, zero, one, zero ), dim=1),
                         torch.cat(( zero, zero, zero, (X_u + X_u_u*2*X[3,0]*torch.sign(X[3,0])).reshape(-1,1)/m11, m22*X[5,0].reshape(-1,1)/m11, m22*X[4,0].reshape(-1,1)/m11, one/m11), dim=1),
                         torch.cat(( zero, zero, zero, -m11*X[5,0].reshape(-1,1)/m22, (Y_v+2*Y_v_v*X[4,0]*torch.sign(X[4,0])).reshape(-1,1)/m22, -m11*X[3,0].reshape(-1,1)/m22, zero) , dim=1),
                         torch.cat(( zero, zero, zero, (m11-m22)*X[4,0].reshape(-1,1)/m33, (m11-m22)*X[3,0].reshape(-1,1)/m33, 1*(N_r+2*N_r_r*X[5,0]*torch.sign(X[5,0])).reshape(-1,1)/m33, zero), dim=1)
                            ))
    
    h = torch.norm( X[0:2] - targetX[0:2] )**2 - min_D**2           
    dh_dx = torch.cat(( 2*( X[0:2] - targetX[0:2] ).T, torch.tensor([[ 0, 0, 0, 0, 0 ]], dtype=torch.float) ), dim=1 )
                    
    # single derivative
    x_dot = f[0,0]
    y_dot = f[1,0]
    dx_dot_dx = torch.cat(( zero, zero, (-ui*torch.sin(phi)-vi*torch.cos(phi)).reshape(-1,1), torch.cos(phi).reshape(-1,1), -torch.sin(phi).reshape(-1,1), zero, zero), dim=1 )
    dy_dot_dx = torch.cat(( zero, zero,  (ui*torch.cos(phi)-vi*torch.sin(phi)).reshape(-1,1), torch.sin(phi).reshape(-1,1), torch.cos(phi).reshape(-1,1), zero, zero ), dim=1 )
    
    h_dot = 2*(xi-targetX[0,0])*x_dot + 2*(yi-targetX[1,0])*y_dot
    dh_dot_dx = 2*(xi-targetX[0,0])*dx_dot_dx + 2*x_dot*torch.tensor([[1,0,0,0,0,0,0]], dtype=torch.float) + 2*(yi-targetX[1,0])*dy_dot_dx + 2*y_dot*torch.tensor([[0,1,0,0,0,0,0]], dtype=torch.float)
    
    # position double derivative
    dx_ddot_dx = Df_dx[3,:] * torch.cos(phi).reshape(-1,1) + f[3,0] * torch.cat((zero, zero, -torch.sin(phi).reshape(-1,1), zero, zero, zero, zero ), dim=1) \
                - torch.cat(( zero, zero, ui*torch.cos(phi).reshape(-1,1), torch.sin(phi).reshape(-1,1), zero, zero, zero ), dim=1) * f[2,0] - ui*torch.sin(phi).reshape(-1,1) * Df_dx[2,:] \
                - Df_dx[4,:] * torch.sin(phi).reshape(-1,1) - f[4,0] * torch.cat(( zero, zero, torch.cos(phi).reshape(-1,1), zero, zero, zero, zero ), dim=1) \
                - torch.cat(( zero, zero, -vi * torch.sin(phi).reshape(-1,1), zero, torch.cos(phi).reshape(-1,1), zero, zero ), dim=1) * f[2,0] - vi*torch.cos(phi).reshape(-1,1) * Df_dx[2,:]
                
    dy_ddot_dx = Df_dx[3,:] * torch.sin(phi).reshape(-1,1) + f[3,0] * torch.cat(( zero, zero, torch.cos(phi).reshape(-1,1), zero, zero, zero, zero ), dim=1) \
                + torch.cat(( zero, zero, -ui*torch.sin(phi).reshape(-1,1), torch.cos(phi).reshape(-1,1), zero, zero, zero ), dim=1) * f[2,0] + ui*torch.cos(phi).reshape(-1,1) * Df_dx[2,:] \
                + Df_dx[4,:] * torch.cos(phi).reshape(-1,1) + f[4,0] * torch.cat(( zero, zero, -torch.sin(phi).reshape(-1,1), zero, zero, zero, zero ), dim=1) \
                - torch.cat(( zero, zero, vi * torch.cos(phi).reshape(-1,1), zero, torch.sin(phi).reshape(-1,1), zero, zero ), dim=1) * f[2,0] - vi*torch.sin(phi).reshape(-1,1) * Df_dx[2,:]
    x_ddot = f[3,0]*torch.cos(phi).reshape(-1,1) - ui*torch.sin(phi).reshape(-1,1)*f[2,0] - f[4,0]*torch.sin(phi).reshape(-1,1) - vi*torch.cos(phi).reshape(-1,1)*f[2,0]
    y_ddot = f[3,0]*torch.sin(phi).reshape(-1,1) + ui*torch.cos(phi).reshape(-1,1)*f[2,0] + f[4,0]*torch.cos(phi).reshape(-1,1) - vi*torch.sin(phi).reshape(-1,1)*f[2,0]
    
    # h double derivative
    h_ddot = 2*torch.linalg.norm( f[0:2] )**2 + 2*(X[0:2]-targetX).T @ torch.cat( (x_ddot, y_ddot), dim=0 ).reshape(-1,1)
    dh_ddot_dx = 4*f[0]*dx_dot_dx + 4*f[1]*dy_dot_dx + 2*(xi-targetX[0,0])*dx_ddot_dx + 2*(yi-targetX[1,0])*dy_ddot_dx + 2*x_ddot * torch.tensor([[1, 0, 0, 0, 0, 0, 0]], dtype=torch.float) + 2*y_ddot*torch.tensor([[0, 1, 0, 0, 0, 0, 0]], dtype=torch.float)

    h2 = h_dot + alpha1 * h
    dh2_dx = dh_dot_dx + alpha1 * dh_dx
    
    h3 = h_ddot + alpha1 * h_dot + alpha2 * ( h2 )
    dh3_dx = dh_ddot_dx + alpha1 * dh_dot_dx + alpha2 * dh2_dx
    
    return h3, dh3_dx
            
def uav2D_lyapunov_jit(X, G):
        
        c1 = 5.0
        c2 = 5.0
        c3 = 1.0
        
        xi = X[0,0]
        yi = X[1,0]
        phi = X[2,0]
        ui = X[3,0]
        vi = X[4,0]
        ri = X[5,0]
        tau_ui = X[6,0]
        
        V_w = 0.0#0.1
        theta_w = torch.tensor(np.pi/3, dtype=torch.float)
        m11 = 5.5404
        m22 = 9.6572
        m33 = 0.0574
        X_u = -2.3015
        X_u_u = -8.2845
        Y_v = -8.0149
        Y_v_v = -23.689
        N_r = -0.0048
        N_r_r = -0.0089
        torch.div( G[1,0]-yi, torch.square(G[0,0]-xi) )
        # V = 20 * torch.linalg.norm( self.X[0:2] - G[0:2] )**2 + torch.linalg.norm( self.X[3:5] )**2
        # dV_dxi = 20* torch.append( 2*( self.X[0:2] - G[0:2] ).T, [[0, 0, 0, 0, 0]] , axis = 1 ) + torch.array([[ 0, 0, 0, 2*self.X[3,0], 2*self.X[4,0], 0, 0 ]])
        
        zero = torch.tensor([[0]], dtype=torch.float)
        one = torch.tensor([[1]], dtype=torch.float)
        
        dist = torch.linalg.norm( torch.linalg.norm(G[0:2]-X[0:2]) )
        if dist<0.05:
            dist = 0.05
        
        theta_g = torch.atan2( G[1,0]-yi, G[0,0]-xi )
        dtheta_g_dx = torch.cos(theta_g)**2 * torch.cat( ( torch.div( G[1,0]-yi, torch.square(G[0,0]-xi) ).reshape(-1,1), torch.div(-one, G[0,0]-xi).reshape(-1,1), zero, zero, zero, zero, zero ), dim=1 )
        
        # tau_d = 
        
        u_di = c1*dist*torch.cos(theta_g-phi)
        
        fu = m22*X[4,0]*X[5,0]+X_u*X[3,0]+X_u_u*torch.abs(X[3,0])*X[3,0]
        dfu_dx = torch.cat( (zero, zero, zero, (X_u + X_u_u*2*X[3,0]*torch.sign(X[3,0])).reshape(-1,1), m22*X[5,0].reshape(-1,1), m22*X[4,0].reshape(-1,1), zero), dim=1)
        tau_uid = -fu + c3*m11*( u_di - ui )
        d_tau_uid_dx = -dfu_dx +  c3*m11*torch.cat( ( (-c1*dist*torch.sin(theta_g-phi)*dtheta_g_dx[0,0]+c1*torch.cos(theta_g-phi)*(-1.0/dist*(G[0,0]-xi))).reshape(-1,1), (-c1*dist*torch.sin(theta_g-phi)*dtheta_g_dx[0,1]+c1*torch.cos(theta_g-phi)*(-1.0/dist*(G[1,0]-yi))).reshape(-1,1), -c1*dist*torch.sin(theta_g-phi).reshape(-1,1), -one, zero, zero, zero), dim=1 )
                       
        Xdi = torch.cat( ( G[0,0].reshape(-1,1), G[1,0].reshape(-1,1), theta_g.reshape(-1,1), c1*torch.linalg.norm( G[0:2]-X[0:2] ) * torch.cos(theta_g-X[2,0]).reshape(-1,1), c1 * torch.linalg.norm( G[0:2]-X[0:2] ) * torch.sin(theta_g-X[2,0]).reshape(-1,1), c2 * (theta_g-X[2,0]).reshape(-1,1), zero ) , dim=0 ) #c3 * ( c1*dist * torch.cos(G[2,0]-self.X[2,0]) - self.X[3,0] ) ] ).reshape(-1,1)
        # Xdi = torch.array( [ G[0,0], G[1,0], G[2,0], c1*torch.linalg.norm( G[0:2]-self.X[0:2] ) * torch.cos(G[2,0]-self.X[2,0]), c1 * torch.linalg.norm( G[0:2]-self.X[0:2] ) * torch.sin(G[2,0]-self.X[2,0]), c2 * (G[2,0]-self.X[2,0]), 0 ] ).reshape(-1,1) #c3 * ( c1*dist * torch.cos(G[2,0]-self.X[2,0]) - self.X[3,0] ) ] ).reshape(-1,1)
        
        V = torch.linalg.norm(X-Xdi)**2     
        
        dV_dx = torch.cat( (  (2*(X[0,0]-Xdi[0,0]) + 2*(phi-theta_g)*( -dtheta_g_dx[0,0] ) + 2*( ui-c1*dist*torch.cos(theta_g-phi) )*(c1*torch.cos(theta_g-phi)/2/dist*2*(G[0,0]-xi)) +  2*( ui-c1*dist*torch.cos(theta_g-phi) )*( c1*dist*torch.sin(theta_g-phi)*dtheta_g_dx[0,0] )  + 2*( vi-c1*dist*torch.sin(theta_g-phi) )*(c1*torch.sin(theta_g-phi)/2/dist*2*(G[0,0]-xi)) + 2*( vi-c1*dist*torch.sin(theta_g-phi) )*( -c1*dist*torch.cos(theta_g-phi)*dtheta_g_dx[0,0] ) + 2*(ri-c2*(theta_g-phi))*(-c2*dtheta_g_dx[0,0]) + 2*(tau_ui-tau_uid)*(-d_tau_uid_dx[0,0])).reshape(-1,1) , 
                             (2*(X[1,0]-Xdi[1,0]) + 2*(phi-theta_g)*( -dtheta_g_dx[0,1] ) + 2*( ui-c1*dist*torch.cos(theta_g-phi) )*(c1*torch.cos(theta_g-phi)/2/dist*2*(G[1,0]-yi)) +  2*( ui-c1*dist*torch.cos(theta_g-phi) )*( c1*dist*torch.sin(theta_g-phi)*dtheta_g_dx[0,1] )  + 2*( vi-c1*dist*torch.sin(theta_g-phi) )*(c1*torch.sin(theta_g-phi)/2/dist*2*(G[1,0]-yi)) + 2*( vi-c1*dist*torch.sin(theta_g-phi) )*( -c1*dist*torch.cos(theta_g-phi)*dtheta_g_dx[0,1] ) + 2*(ri-c2*(theta_g-phi))*(-c2*dtheta_g_dx[0,1]) + 2*(tau_ui-tau_uid)*(-d_tau_uid_dx[0,1])).reshape(-1,1),
                             (2*( phi-theta_g ) + 2*( ui-c1*dist*torch.cos(theta_g-phi) )*( -c1*dist*(torch.sin(theta_g-phi)) ) + 2*( vi-c1*dist*torch.sin(theta_g-phi) )*( -c1*dist*(-torch.cos(theta_g-phi)) ) + 2*( ri-c2*(theta_g-phi) )*( c2 ) + 2*(tau_ui-tau_uid)*(-d_tau_uid_dx[0,2])).reshape(-1,1),
                             (2*( ui-c1*dist*torch.cos(theta_g-phi) ) + 2*(tau_ui-tau_uid)*(-d_tau_uid_dx[0,3])).reshape(-1,1),
                             (2*( vi-c1*dist*torch.sin(theta_g-phi) ) + 2*(tau_ui-tau_uid)*(-d_tau_uid_dx[0,4])).reshape(-1,1),
                             (2*( ri-c2*(theta_g-phi) ) + 2*(tau_ui-tau_uid)*(-d_tau_uid_dx[0,5])).reshape(-1,1),
                             (2*(tau_ui-tau_uid) + 2*(tau_ui-tau_uid)*(-d_tau_uid_dx[0,6])).reshape(-1,1)  ), dim=1 )    
        
        # dV_dx = torch.array([[  2*(self.X[0,0]-Xdi[0,0]) + 2*( ui-c1*dist*torch.cos(G[2,0]-phi) )/2/dist*2*(self.X[0,0]-xi) + 2*( vi-c1*dist*torch.sin(G[2,0]-phi) )/2/dist*2*(self.X[0,0]-xi)  , 
        #                      2*(self.X[1,0]-Xdi[1,0]) + 2*( ui-c1*dist*torch.cos(G[2,0]-phi) )/2/dist*2*(self.X[1,0]-yi) + 2*( vi-c1*dist*torch.sin(G[2,0]-phi) )/2/dist*2*(self.X[1,0]-yi),
        #                      2*( phi-G[2,0] ) + 2*( ui-c1*dist*torch.cos(G[2,0]-phi) )*( -c1*dist*(torch.sin(G[2,0]-phi)) ) + 2*( vi-c1*dist*torch.sin(G[2,0]-phi) )*( -c1*dist*(-torch.cos(G[2,0]-phi)) ) + 2*( ri-c2*(G[2,0]-phi) )*( c2 ),
        #                      2*( ui-c1*dist*torch.cos(G[2,0]-phi) ),
        #                      2*( vi-c1*dist*torch.sin(G[2,0]-phi) ),
        #                      2*( ri-c2*(G[2,0]-phi) ),
        #                      2*tau_ui  ]])        
        
        return V, dV_dx


    
# @torch.jit.script
def uav_compute_reward_jit(X,targetX, control):
    # return 100 * torch.square( torch.norm( X[0:2,0] - targetX[0:2,0]  ) ) #- torch.tensor((min_D+max_D)/2) ) - 2 * h3
    return 100 * torch.square( torch.norm( X[0:2,0] - targetX[0:2,0]  ) ) + torch.square( torch.norm(control) )
traced_uav_compute_reward_jit = torch.jit.trace( uav_compute_reward_jit, ( torch.ones(7,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float) ) )
    
def uav2D_qp_constraints_jit(state, goal, obs1, obs2, param0, param1, param2, param3): #k, alpha1, alpha2, alpha3
    
    V, dV_dx = uav2D_lyapunov_jit( state, goal )
    h1, dh1_dx = uav_SI2D_barrier_jit( state, obs1, param1, param2  )
    h2, dh2_dx = uav_SI2D_barrier_jit( state, obs2, param1, param2  )
    f = uav_f_torch_jit(state)
    g = uav_g_torch_jit(state)
    A0 = -dV_dx @ g; b0 = -dV_dx @ f - param0 * V
    A1 = dh1_dx @ g; b1 = dh1_dx @ f + param1 * h1
    A2 = dh2_dx @ g; b2 = dh2_dx @ f + param2 * h2        
    A = torch.cat( (A0, A1, A2), dim=0 )
    b = torch.cat( (b0, b1, b2), dim=0 )
    # A = torch.cat( (A0, A1), dim=0 )
    # b = torch.cat( (b0, b1), dim=0 )
    return A, b
traced_uav2D_qp_constraints_jit = torch.jit.trace( uav2D_qp_constraints_jit, ( torch.ones(7,1, dtype=torch.float), torch.zeros(2,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.ones(2,1, dtype=torch.float), torch.tensor(0.5, dtype=torch.float), torch.tensor(0.5, dtype=torch.float), torch.tensor(0.5, dtype=torch.float), torch.tensor(0.5, dtype=torch.float) ) )
