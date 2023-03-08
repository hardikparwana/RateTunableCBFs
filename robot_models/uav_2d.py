import numpy as np
from utils.utils import wrap_angle
from trust_utils import *
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cvxpy as cp
from robot_models.obstacles import circle2D

class UAV_2d:
    
    def __init__(self,X0,dt,ax,id = 0, mode = 'ego', target = 0, color='r',alpha = 0.8, palpha=1.0,plot=True, nominal_plot=True, num_constraints = 0, num_robots = 1, alpha1 = 0.8, alpha2 = 0.8):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'Unicycle2D'
        
        self.X = X0.reshape(-1,1)
        self.X_nominal = np.copy(self.X)
        self.dt = dt
        self.target = target
        self.mode = mode
        
        self.id = id
        self.color = color
        self.palpha = palpha
        self.min_D = 0.3
        self.max_D = 2.0
        
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
        self.U = np.array([0,0]).reshape(-1,1)
        
        # Plot handles
        self.plot = plot
        self.plot_nominal = nominal_plot
        if self.plot:
            self.body = ax.scatter([],[],alpha=palpha,s=60,facecolors=self.color,edgecolors=self.color) #facecolors='none'
            self.radii = 0.4
            self.palpha = palpha
            if palpha==1:
                self.axis = ax.plot([self.X[0,0],self.X[0,0]+self.radii*np.cos(self.X[2,0])],[self.X[1,0],self.X[1,0]+self.radii*np.sin(self.X[2,0])], color=self.color)
            self.render_plot()
            
        if self.plot_nominal:
            self.body_nominal = ax.scatter([],[],alpha=0.3,s=60,facecolors=self.color,edgecolors=self.color) #facecolors='none'
            if palpha==1:
                self.axis_nominal = ax.plot([self.X_nominal[0,0],self.X_nominal[0,0]+self.radii*np.cos(self.X_nominal[2,0])],[self.X_nominal[1,0],self.X_nominal[1,0]+self.radii*np.sin(self.X_nominal[2,0])], alpha = 0.3, color=self.color)
            self.render_plot_nominal()
            
        self.Xs = np.copy(self.X)
        self.Us = np.copy(self.U)
        
        # to store constraints
        self.A = np.zeros((num_constraints,2))
        self.b = np.zeros((num_constraints,1))
        self.agent_objective = [0] * num_robots
        self.U_ref = np.array([0,0]).reshape(-1,1)
        self.U_nominal = np.array([0,0]).reshape(-1,1)
        self.alpha = alpha * np.ones((1,num_robots))
        
        # trust
        self.trust = np.zeros((1,num_robots))
        
    def f(self):
        V_w = 0.1
        theta_w = np.pi/3
        m11 = 5.5404
        m22 = 9.6572
        m33 = 0.0574
        X_u = -2.3015
        X_u_u = -8.2845
        Y_v = -8.0149
        Y_v_v = -23.689
        N_r = -0.0048
        N_r_r = -0.0089
        return np.array([self.X[3,0]*np.cos(self.X[2,0])-self.X[4,0]*np.sin(self.X[2,0])+V_w*np.cos(theta_w),
                       self.X[3,0]*np.sin(self.X[2,0])+self.X[4,0]*np.cos(self.X[2,0])+V_w*np.sin(theta_w),
                       self.X[5,0],
                       (m22*self.X[4,0]*self.X[5,0]+X_u*self.X[3,0]+X_u_u*np.abs(self.X[3,0])*self.X[3,0]+self.X[6,0])/m11,
                       (-m11*self.X[3,0]*self.X[5,0]+Y_v*self.X[4,0]+Y_v_v*np.abs(self.X[4,0])*self.X[4,0])/m22,
                       ((m11-m22)*self.X[3,0]*self.X[4,0]+N_r*self.X[5,0]+N_r_r*np.abs(self.X[5,0])*self.X[5,0])/m33,
                       0]).reshape(-1,1)
    
    def g(self):
        V_w = 0.1
        theta_w = np.pi/3
        m11 = 5.5404
        m22 = 9.6572
        m33 = 0.0574
        X_u = -2.3015
        X_u_u = -8.2845
        Y_v = -8.0149
        Y_v_v = -23.689
        N_r = -0.0048
        N_r_r = -0.0089
        return np.array([ [ 0, 0 ],
                          [ 0, 0 ],
                          [ 0, 0 ],
                          [ 0, 0 ],
                          [ 0, 0 ],
                          [ 0, 1/m33 ],
                          [ 1, 0 ] ])  
        
    def f_nominal(self):
        V_w = 0.1
        theta_w = np.pi/3
        m11 = 5.5404
        m22 = 9.6572
        m33 = 0.0574
        X_u = -2.3015
        X_u_u = -8.2845
        Y_v = -8.0149
        Y_v_v = -23.689
        N_r = -0.0048
        N_r_r = -0.0089
        return np.array([self.X_nominal[3,0]*np.cos(self.X_nominal[2,0])-self.X_nominal[4,0]*np.sin(self.X_nominal[2,0])+V_w*np.cos(theta_w),
                       self.X_nominal[3,0]*np.sin(self.X_nominal[2,0])+self.X_nominal[4,0]*np.cos(self.X_nominal[2,0])+V_w*np.sin(theta_w),
                       self.X_nominal[5,0],
                       (m22*self.X_nominal[4,0]*self.X_nominal[5,0]+X_u*self.X_nominal[3,0]+X_u_u*np.abs(self.X_nominal[3,0])*self.X_nominal[3,0]+self.X[6,0])/m11,
                       (-m11*self.X_nominal[3,0]*self.X_nominal[5,0]+Y_v*self.X_nominal[4,0]+Y_v_v*np.abs(self.X_nominal[4,0])*self.X_nominal[4,0])/m22,
                       ((m11-m22)*self.X_nominal[3,0]*self.X_nominal[4,0]+N_r*self.X_nominal[5,0]+N_r_r*np.abs(self.X_nominal[5,0])*self.X_nominal[5,0])/m33,
                       0 ] ).reshape(-1,1)
    
    def g_nominal(self):
        V_w = 0.1
        theta_w = np.pi/3
        m11 = 5.5404
        m22 = 9.6572
        m33 = 0.0574
        X_u = -2.3015
        X_u_u = -8.2845
        Y_v = -8.0149
        Y_v_v = -23.689
        N_r = -0.0048
        N_r_r = -0.0089
        return np.array([ [ 0, 0 ],
                          [ 0, 0 ],
                          [ 0, 0 ],
                          [ 0, 0 ],
                          [ 0, 0 ],
                          [ 0, 1/m33 ],
                          [ 1, 0 ] ])  
        
    def Xdot(self):
        return self.f() + self.g() @ self.U     
         
    def step(self,U): 
        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        self.X[2,0] = wrap_angle(self.X[2,0])
        if self.plot:
            self.render_plot()
        self.Xs = np.append(self.Xs,self.X,axis=1)
        self.Us = np.append(self.Us,self.U,axis=1)
        return self.X
    
    def step_nominal(self,U): 
        self.U_nominal = U.reshape(-1,1)
        self.X_nominal = self.X_nominal + ( self.f_nominal() + self.g_nominal() @ self.U_nominal )*self.dt
        self.X_nominal[2,0] = wrap_angle(self.X_nominal[2,0])
        self.render_plot_nominal()
        return self.X_nominal
    
    def render_plot(self):
        if self.plot:
            x = np.array([self.X[0,0],self.X[1,0]])
            self.body.set_offsets([x[0],x[1]])
            
            if self.palpha==1:
                self.axis[0].set_ydata([self.X[1,0],self.X[1,0]+self.radii*np.sin(self.X[2,0])])
                self.axis[0].set_xdata( [self.X[0,0],self.X[0,0]+self.radii*np.cos(self.X[2,0])] )
    
    def render_plot_nominal(self):
        if self.plot_nominal:
            x = np.array([self.X_nominal[0,0],self.X_nominal[1,0]])
            self.body_nominal.set_offsets([x[0],x[1]])
            
            if self.palpha==1:
                self.axis_nominal[0].set_ydata([self.X_nominal[1,0],self.X_nominal[1,0]+self.radii*np.sin(self.X_nominal[2,0])])
                self.axis_nominal[0].set_xdata( [self.X_nominal[0,0],self.X_nominal[0,0]+self.radii*np.cos(self.X_nominal[2,0])] )
        
    def lyapunov(self, G):
        
        c1 = 5.0
        c2 = 5.0
        c3 = 1.0
        
        V_w = 0.1
        theta_w = np.pi/3
        m11 = 5.5404
        m22 = 9.6572
        m33 = 0.0574
        X_u = -2.3015
        X_u_u = -8.2845
        Y_v = -8.0149
        Y_v_v = -23.689
        N_r = -0.0048
        N_r_r = -0.0089
        
        # V = 20 * np.linalg.norm( self.X[0:2] - G[0:2] )**2 + np.linalg.norm( self.X[3:5] )**2
        # dV_dxi = 20* np.append( 2*( self.X[0:2] - G[0:2] ).T, [[0, 0, 0, 0, 0]] , axis = 1 ) + np.array([[ 0, 0, 0, 2*self.X[3,0], 2*self.X[4,0], 0, 0 ]])
        
        xi = self.X[0,0]
        yi = self.X[1,0]
        phi = self.X[2,0]
        ui = self.X[3,0]
        vi = self.X[4,0]
        ri = self.X[5,0]
        tau_ui = self.X[6,0]
        
        dist = np.linalg.norm( np.linalg.norm(G[0:2]-self.X[0:2]) )
        if dist<0.05:
            dist = 0.05
        
        theta_g = np.arctan2( G[1,0]-yi, G[0,0]-xi )
        dtheta_g_dx = np.cos(theta_g)**2 * np.array([[ (G[1,0]-yi)/(G[0,0]-xi)**2, -1/(G[0,0]-xi), 0, 0, 0, 0, 0 ]])
        
        # tau_d = 
        
        u_di = c1*dist*np.cos(theta_g-phi)
        
        fu = m22*self.X[4,0]*self.X[5,0]+X_u*self.X[3,0]+X_u_u*np.abs(self.X[3,0])*self.X[3,0]
        dfu_dx = np.array([[ 0, 0, 0, (X_u + X_u_u*2*self.X[3,0]*np.sign(self.X[3,0])), m22*self.X[5,0], m22*self.X[4,0], 0 ]])
        tau_uid = -fu + c3*m11*( u_di - ui )
        d_tau_uid_dx = -dfu_dx +  c3*m11*np.array([[ -c1*dist*np.sin(theta_g-phi)*dtheta_g_dx[0,0]+c1*np.cos(theta_g-phi)*(-1.0/dist*(G[0,0]-xi)), -c1*dist*np.sin(theta_g-phi)*dtheta_g_dx[0,1]+c1*np.cos(theta_g-phi)*(-1.0/dist*(G[1,0]-yi)), -c1*dist*np.sin(theta_g-phi), -1, 0, 0, 0   ]])
                       
        Xdi = np.array( [ G[0,0], G[1,0], theta_g, c1*np.linalg.norm( G[0:2]-self.X[0:2] ) * np.cos(theta_g-self.X[2,0]), c1 * np.linalg.norm( G[0:2]-self.X[0:2] ) * np.sin(theta_g-self.X[2,0]), c2 * (theta_g-self.X[2,0]), 0 ] ).reshape(-1,1) #c3 * ( c1*dist * np.cos(G[2,0]-self.X[2,0]) - self.X[3,0] ) ] ).reshape(-1,1)
        # Xdi = np.array( [ G[0,0], G[1,0], G[2,0], c1*np.linalg.norm( G[0:2]-self.X[0:2] ) * np.cos(G[2,0]-self.X[2,0]), c1 * np.linalg.norm( G[0:2]-self.X[0:2] ) * np.sin(G[2,0]-self.X[2,0]), c2 * (G[2,0]-self.X[2,0]), 0 ] ).reshape(-1,1) #c3 * ( c1*dist * np.cos(G[2,0]-self.X[2,0]) - self.X[3,0] ) ] ).reshape(-1,1)
        
        V = np.linalg.norm(self.X-Xdi)**2     
        
        dV_dx = np.array([[  2*(self.X[0,0]-Xdi[0,0]) + 2*(phi-theta_g)*( -dtheta_g_dx[0,0] ) + 2*( ui-c1*dist*np.cos(theta_g-phi) )*(c1*np.cos(theta_g-phi)/2/dist*2*(G[0,0]-xi)) +  2*( ui-c1*dist*np.cos(theta_g-phi) )*( c1*dist*np.sin(theta_g-phi)*dtheta_g_dx[0,0] )  + 2*( vi-c1*dist*np.sin(theta_g-phi) )*(c1*np.sin(theta_g-phi)/2/dist*2*(G[0,0]-xi)) + 2*( vi-c1*dist*np.sin(theta_g-phi) )*( -c1*dist*np.cos(theta_g-phi)*dtheta_g_dx[0,0] ) + 2*(ri-c2*(theta_g-phi))*(-c2*dtheta_g_dx[0,0]) + 2*(tau_ui-tau_uid)*(-d_tau_uid_dx[0,0]) , 
                             2*(self.X[1,0]-Xdi[1,0]) + 2*(phi-theta_g)*( -dtheta_g_dx[0,1] ) + 2*( ui-c1*dist*np.cos(theta_g-phi) )*(c1*np.cos(theta_g-phi)/2/dist*2*(G[1,0]-yi)) +  2*( ui-c1*dist*np.cos(theta_g-phi) )*( c1*dist*np.sin(theta_g-phi)*dtheta_g_dx[0,1] )  + 2*( vi-c1*dist*np.sin(theta_g-phi) )*(c1*np.sin(theta_g-phi)/2/dist*2*(G[1,0]-yi)) + 2*( vi-c1*dist*np.sin(theta_g-phi) )*( -c1*dist*np.cos(theta_g-phi)*dtheta_g_dx[0,1] ) + 2*(ri-c2*(theta_g-phi))*(-c2*dtheta_g_dx[0,1]) + 2*(tau_ui-tau_uid)*(-d_tau_uid_dx[0,1]),
                             2*( phi-theta_g ) + 2*( ui-c1*dist*np.cos(theta_g-phi) )*( -c1*dist*(np.sin(theta_g-phi)) ) + 2*( vi-c1*dist*np.sin(theta_g-phi) )*( -c1*dist*(-np.cos(theta_g-phi)) ) + 2*( ri-c2*(theta_g-phi) )*( c2 ) + 2*(tau_ui-tau_uid)*(-d_tau_uid_dx[0,2]),
                             2*( ui-c1*dist*np.cos(theta_g-phi) ) + 2*(tau_ui-tau_uid)*(-d_tau_uid_dx[0,3]),
                             2*( vi-c1*dist*np.sin(theta_g-phi) ) + 2*(tau_ui-tau_uid)*(-d_tau_uid_dx[0,4]),
                             2*( ri-c2*(theta_g-phi) ) + 2*(tau_ui-tau_uid)*(-d_tau_uid_dx[0,5]),
                             2*(tau_ui-tau_uid) + 2*(tau_ui-tau_uid)*(-d_tau_uid_dx[0,6])  ]])    
        
        # dV_dx = np.array([[  2*(self.X[0,0]-Xdi[0,0]) + 2*( ui-c1*dist*np.cos(G[2,0]-phi) )/2/dist*2*(self.X[0,0]-xi) + 2*( vi-c1*dist*np.sin(G[2,0]-phi) )/2/dist*2*(self.X[0,0]-xi)  , 
        #                      2*(self.X[1,0]-Xdi[1,0]) + 2*( ui-c1*dist*np.cos(G[2,0]-phi) )/2/dist*2*(self.X[1,0]-yi) + 2*( vi-c1*dist*np.sin(G[2,0]-phi) )/2/dist*2*(self.X[1,0]-yi),
        #                      2*( phi-G[2,0] ) + 2*( ui-c1*dist*np.cos(G[2,0]-phi) )*( -c1*dist*(np.sin(G[2,0]-phi)) ) + 2*( vi-c1*dist*np.sin(G[2,0]-phi) )*( -c1*dist*(-np.cos(G[2,0]-phi)) ) + 2*( ri-c2*(G[2,0]-phi) )*( c2 ),
        #                      2*( ui-c1*dist*np.cos(G[2,0]-phi) ),
        #                      2*( vi-c1*dist*np.sin(G[2,0]-phi) ),
        #                      2*( ri-c2*(G[2,0]-phi) ),
        #                      2*tau_ui  ]])        
        
        return V, dV_dx
    
    def nominal_input(self,G, type, d_min = 0.3):
        G = np.copy(G.reshape(-1,1))
        k_omega = 2.0 #0.5#2.5
        k_v = 3.0 #2.0 #0.5
        distance = max(np.linalg.norm( self.X[0:2,0]-G[0:2,0] ) - d_min,0)
        theta_d = np.arctan2(G[1,0]-self.X[1,0],G[0,0]-self.X[0,0])
        error_theta = wrap_angle( theta_d - self.X[2,0] )

        omega = k_omega*error_theta   
        v = k_v*( distance )*np.cos( error_theta )
        return np.array([v, omega]).reshape(-1,1)
       
    def agent_barrier(self, agent, d_min):
        
        V_w = 0.1
        theta_w = np.pi/3
        m11 = 5.5404
        m22 = 9.6572
        m33 = 0.0574
        X_u = -2.3015
        X_u_u = -8.2845
        Y_v = -8.0149
        Y_v_v = -23.689
        N_r = -0.0048
        N_r_r = -0.0089
        
        xi = self.X[0,0]
        yi = self.X[1,0]
        phi = self.X[2,0]
        ui = self.X[3,0]
        vi = self.X[4,0]
        ri = self.X[5,0]
        
        f = np.array([self.X[3,0]*np.cos(self.X[2,0])-self.X[4,0]*np.sin(self.X[2,0])+V_w*np.cos(theta_w),
                       self.X[3,0]*np.sin(self.X[2,0])+self.X[4,0]*np.cos(self.X[2,0])+V_w*np.sin(theta_w),
                       self.X[5,0],
                       (m22*self.X[4,0]*self.X[5,0]+X_u*self.X[3,0]+X_u_u*np.abs(self.X[3,0])*self.X[3,0]+self.X[6,0])/m11,
                       (-m11*self.X[3,0]*self.X[5,0]+Y_v*self.X[4,0]+Y_v_v*np.abs(self.X[4,0])*self.X[4,0])/m22,
                       ((m11-m22)*self.X[3,0]*self.X[4,0]+N_r*self.X[5,0]+N_r_r*np.abs(self.X[5,0])*self.X[5,0])/m33,
                       0]).reshape(-1,1)
        
        Df_dx = np.array([ 
                         [ 0, 0, -self.X[3,0]*np.sin(self.X[2,0])-self.X[4,0]*np.cos(self.X[2,0]), np.cos(self.X[2,0]), -np.sin(self.X[2,0]), 0, 0],
                         [ 0, 0, self.X[3,0]*np.cos(self.X[2,0])-self.X[4,0]*np.sin(self.X[2,0]), np.sin(self.X[2,0]), np.cos(self.X[2,0]), 0, 0],
                         [ 0, 0, 0, 0, 0, 1, 0 ],
                         [ 0, 0, 0, (X_u + X_u_u*2*self.X[3,0]*np.sign(self.X[3,0]))/m11, m22*self.X[5,0]/m11, m22*self.X[4,0]/m11, 1/m11 ],
                         [ 0, 0, 0, -m11*self.X[5,0]/m22, (Y_v+2*Y_v_v*self.X[4,0]*np.sign(self.X[4,0]))/m22, -m11*self.X[3,0]/m22, 0  ],
                         [ 0, 0, 0, (m11-m22)*self.X[4,0]/m33, (m11-m22)*self.X[3,0]/m33, 2*(N_r+N_r_r*self.X[5,0]*np.sign(self.X[5,0]))/m33, 0 ]
                        ])
        
        h = np.linalg.norm( self.X[0:2] - agent.X[0:2] )**2 - d_min**2           
        dh_dx = np.append( 2*( self.X[0:2] - agent.X[0:2] ).T, [[ 0, 0, 0, 0, 0 ]], axis=1 )
                        
        # single derivative
        x_dot = f[0,0]
        y_dot = f[1,0]
        dx_dot_dx = np.array([ 0, 0, -ui*np.sin(phi)-vi*np.cos(phi), np.cos(phi), -np.sin(phi), 0, 0  ])
        dy_dot_dx = np.array([ 0, 0, ui*np.cos(phi)-vi*np.sin(phi), np.sin(phi), np.cos(phi), 0, 0 ])
        
        h_dot = 2*(xi-agent.X[0,0])*x_dot + 2*(yi-agent.X[1,0])*y_dot
        dh_dot_dx = 2*(xi-agent.X[0,0])*dx_dot_dx + 2*x_dot*np.array([[1,0,0,0,0,0,0]]) + 2*(yi-agent.X[1,0])*dy_dot_dx + 2*y_dot*np.array([[0,1,0,0,0,0,0]])
        
        # position double derivative
        dx_ddot_dx = Df_dx[3,:] * np.cos(phi) + f[3,0] * np.array([[ 0, 0, -np.sin(phi), 0, 0, 0, 0 ]]) \
                    - np.array([ 0, 0, ui*np.cos(phi), np.sin(phi), 0, 0, 0 ]) * f[2,0] - ui*np.sin(phi) * Df_dx[2,:] \
                    - Df_dx[4,:] * np.sin(phi) - f[4,0] * np.array([[ 0, 0, np.cos(phi), 0, 0, 0, 0 ]]) \
                    - np.array([[ 0, 0, -vi * np.sin(phi), 0, np.cos(phi), 0, 0 ]]) * f[2,0] - vi*np.cos(phi) * Df_dx[2,:]
                    
        dy_ddot_dx = Df_dx[3,:] * np.sin(phi) + f[3,0] * np.array([[ 0, 0, np.cos(phi), 0, 0, 0, 0 ]]) \
                    + np.array([ 0, 0, -ui*np.sin(phi), np.cos(phi), 0, 0, 0 ]) * f[2,0] + ui*np.cos(phi) * Df_dx[2,:] \
                    + Df_dx[4,:] * np.cos(phi) + f[4,0] * np.array([[ 0, 0, -np.sin(phi), 0, 0, 0, 0 ]]) \
                    - np.array([[ 0, 0, vi * np.cos(phi), 0, np.sin(phi), 0, 0 ]]) * f[2,0] - vi*np.sin(phi) * Df_dx[2,:]
        x_ddot = f[3,0]*np.cos(phi) - ui*np.sin(phi)*f[2,0] - f[4,0]*np.sin(phi) - vi*np.cos(phi)*f[2,0]
        y_ddot = f[3,0]*np.sin(phi) + ui*np.cos(phi)*f[2,0] + f[4,0]*np.cos(phi) - vi*np.sin(phi)*f[2,0]
        
        # h double derivative
        h_ddot = 2*np.linalg.norm( f[0:2] )**2 + 2*(self.X[0:2]-agent.X).T @ np.append( x_ddot, y_ddot ).reshape(-1,1)
        dh_ddot_dx = 4*f[0]*dx_dot_dx + 4*f[1]*dy_dot_dx + 2*(xi-agent.X[0,0])*dx_ddot_dx + 2*(yi-agent.X[1,0])*dy_ddot_dx + 2*x_ddot * np.array([[1, 0, 0, 0, 0, 0, 0]]) + 2*y_ddot*np.array([[0, 1, 0, 0, 0, 0, 0]])
    
        h2 = h_dot + self.alpha1 * h
        dh2_dx = dh_dot_dx + self.alpha1 * dh_dx
        
        h3 = h_ddot + self.alpha1 * h_dot + self.alpha2 * ( h2 )
        dh3_dx = dh_ddot_dx + self.alpha1 * dh_dot_dx + self.alpha2 * dh2_dx
            
        # assert(h2>=0)
        # assert(h3>=0)
        
        return h3, dh3_dx


if 0:
    plt.ion()
    fig = plt.figure()
    ax = plt.axes( xlim = (-2,2), ylim = (-2,2) )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect(1)
    
    dt = 0.01
    tf = 20
    num_steps = int(tf/dt)
    alpha3 = 0.8
    robot = UAV_2d( np.array([0,0,0,0.1,0.1,0,0.1]).reshape(-1,1), dt, ax, alpha1 = 2.0, alpha2 = 2.0 )
    obsX = [0.5, 0.5]
    obs2X = [1.5, 1.9]
    targetX = np.array([1, 1]).reshape(-1,1)
    d_min = 0.3
    obs1 = circle2D(obsX[0], obsX[1], d_min, ax, 0)
    obs2 = circle2D(obs2X[0], obs2X[1], d_min, ax, 0)
    
    ax.scatter(targetX[0,0], targetX[1,0],c='g')
    
    num_constraints = 2
    u = cp.Variable((2,1))
    u_ref = cp.Parameter((2,1), value = np.zeros((2,1)))
    A1 = cp.Parameter((num_constraints,2), value = np.zeros((num_constraints,2)))
    b1 = cp.Parameter((num_constraints,1), value = np.zeros((num_constraints,1)))
    delta = cp.Variable((num_constraints,1), value = np.zeros((num_constraints,1)))
    objective = cp.Minimize( cp.sum_squares( u - u_ref ) + 10 * cp.sum_squares( delta[0,0] ) + 100000 * cp.sum_squares(delta[1:,:]) ) 
    # factor_matrix = np.zeros((num_constraints,1)); factor_matrix[0,0] = 1
    const = [A1 @ u + b1 + delta >= 0]
    const += [ cp.abs( u[0,0] ) <= 5 ]
    const += [ cp.abs( u[1,0] ) <= 2.0 ]
    cbf_controller = cp.Problem( objective, const )
    
    for i in range(num_steps):
        
        h3, dh3_dx = robot.agent_barrier( obs1, d_min )
        h4, dh4_dx = robot.agent_barrier( obs2, d_min )
        V, dV_dx = robot.lyapunov( targetX )
        # print(f"V:{V}, dV_dx:{dV_dx}")
        A1.value[0,:] = 0.0*(-dV_dx @ robot.g())
        A1.value[1,:] = 1.0*(dh3_dx @ robot.g())
        b1.value[0,:] = 0.0*(-dV_dx @ robot.f() - 1.0 * V )
        b1.value[1,:] = 1.0*(dh3_dx @ robot.f() + alpha3 * h3)
        cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)
        
        if cbf_controller.status!='optimal':
            print("ERROR in QP")
                  
        print(f"control input: {u.value.T}")
        robot.step(u.value)
        # print(f"ui:{robot.X[3,0]}, vi:{robot.X[4,0]}")
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        