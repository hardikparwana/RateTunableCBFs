import numpy as np
from utils.utils import wrap_angle

class Unicycle2D:
    
    def __init__(self,X0,dt,ax,id = 0, mode = 'ego', target = 0, color='r',alpha = 0.8, palpha=1.0,plot=True, nominal_plot=True, num_constraints = 0, num_robots = 1):
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
        
        self.U = np.array([0,0]).reshape(-1,1)
        
        # Plot handles
        self.plot = plot
        self.plot_nominal = nominal_plot
        if self.plot:
            self.body = ax.scatter([],[],[],alpha=palpha,s=60,facecolors=self.color,edgecolors=self.color) #facecolors='none'
            self.radii = 1.0
            self.palpha = palpha
            if palpha==1:
                self.axis = ax.plot([self.X[0,0],self.X[0,0]+self.radii*np.cos(self.X[3,0])],[self.X[1,0],self.X[1,0]+self.radii*np.sin(self.X[3,0])],[0,0], color=self.color)
            self.render_plot()
        if self.plot_nominal:
            self.body_nominal = ax.scatter([],[],[],alpha=0.3,s=60,facecolors=self.color,edgecolors=self.color) #facecolors='none'
            if palpha==1:
                self.axis_nominal = ax.plot([self.X_nominal[0,0],self.X_nominal[0,0]+self.radii*np.cos(self.X_nominal[3,0])],[self.X_nominal[1,0],self.X_nominal[1,0]+self.radii*np.sin(self.X_nominal[3,0])],[0,0], alpha = 0.3, color=self.color)
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
        
    def f(self):
        return np.array([0,0,0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [ np.cos(self.X[3,0]), 0],
                          [ np.sin(self.X[3,0]), 0],
                          [ 0, 0],
                          [0, 1] ])  
        
    def f_nominal(self):
        return np.array([0,0,0,0]).reshape(-1,1)
    
    def g_nominal(self):
        return np.array([ [ np.cos(self.X_nominal[3,0]), 0 ],
                          [ np.sin(self.X_nominal[3,0]), 0],
                          [ 0, 0],
                          [0, 1] ])  
        
    def Xdot(self):
        return self.f() + self.g() @ self.U     
         
    def step(self,U): 
        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        self.X[3,0] = wrap_angle(self.X[3,0])
        self.render_plot()
        self.Xs = np.append(self.Xs,self.X,axis=1)
        self.Us = np.append(self.Us,self.U,axis=1)
        return self.X
    
    def step_nominal(self,U): 
        self.U_nominal = U.reshape(-1,1)
        self.X_nominal = self.X_nominal + ( self.f_nominal() + self.g_nominal() @ self.U_nominal )*self.dt
        self.X_nominal[3,0] = wrap_angle(self.X_nominal[3,0])
        self.render_plot_nominal()
        return self.X_nominal
    
    def render_plot(self):
        if self.plot:
            x = np.array([self.X[0,0],self.X[1,0],self.X[2,0]])
            self.body._offsets3d = ([[x[0]],[x[1]],[x[2]]])
            
            if self.palpha==1:
                self.axis[0].set_ydata([self.X[1,0],self.X[1,0]+self.radii*np.sin(self.X[3,0])])
                self.axis[0].set_xdata( [self.X[0,0],self.X[0,0]+self.radii*np.cos(self.X[3,0])] )
                self.axis[0].set_3d_properties( [self.X[2,0],self.X[2,0]] )
        # self.axis = ax.plot([self.X[0,0],self.X[0,0]+np.cos(self.X[2,0])],[self.X[1,0],self.X[1,0]+np.sin(self.X[2,0])])
    
    def render_plot_nominal(self):
        if self.plot_nominal:
            x = np.array([self.X_nominal[0,0],self.X_nominal[1,0],self.X_nominal[2,0]])
            self.body_nominal._offsets3d = ([[x[0]],[x[1]],[x[2]]])
            
            if self.palpha==1:
                self.axis_nominal[0].set_ydata([self.X_nominal[1,0],self.X_nominal[1,0]+self.radii*np.sin(self.X_nominal[3,0])])
                self.axis_nominal[0].set_xdata( [self.X_nominal[0,0],self.X_nominal[0,0]+self.radii*np.cos(self.X_nominal[3,0])] )
                self.axis_nominal[0].set_3d_properties( [self.X_nominal[2,0],self.X_nominal[2,0]] )
    
    def lyapunov(self, G, type='none'):
        V = np.linalg.norm( self.X[0:3] - G[0:3] )**2
        dV_dxi = np.append( 2*( self.X[0:3] - G[0:3] ).T, [[0]] , axis = 1 )
        
        if type=='SingleIntegrator3D':
            dV_dxj = -2*( self.X[0:3] - G[0:3] ).T
        elif type=='SingleIntegrator6D':
            dV_dxj = np.append( -2*( self.X[0:3] - G[0:3] ).T, [[0, 0, 0]], axis = 1 )
        elif type=='Unicycle2D':
            dV_dxj = np.append( -2*( self.X[0:3] - G[0:3] ).T, [[0]], axis=1  )
        elif type=='DoubleIntegrator3D':
            dV_dxj = np.append( -2*( self.X[0:3] - G[0:3] ).T, [[0, 0, 0]], axis = 1 )
        else:
            dV_dxj = -2*( self.X[0:3] - G[0:3] ).T
        
        return V, dV_dxi, dV_dxj
    
    def nominal_input(self,G, type, d_min = 0.3):
        G = np.copy(G.reshape(-1,1))
        k_omega = 2.0 #0.5#2.5
        k_v = 3.0 #2.0 #0.5
        distance = max(np.linalg.norm( self.X[0:2,0]-G[0:2,0] ) - d_min,0)
        theta_d = np.arctan2(G[1,0]-self.X[1,0],G[0,0]-self.X[0,0])
        error_theta = wrap_angle( theta_d - self.X[3,0] )

        omega = k_omega*error_theta   
        v = k_v*( distance )*np.cos( error_theta )
        return np.array([v, omega]).reshape(-1,1)
    
    def sigma(self,s):
        k1 = 0.5
        k2 = 4.0
        return k2 * (np.exp(k1-s)-1)/(np.exp(k1-s)+1)
    
    def sigma_der(self,s):
        k1 = 0.5
        k2 = 4.0    
        return - k2 * np.exp(k1-s)/( 1+np.exp( k1-s ) ) * ( 1 - self.sigma(s)/k2 )
    
    def agent_barrier(self, agent, d_min):
        
        beta = 1.01
        theta = self.X[3,0]
        
        if agent.type!='Surveillance':
        
            h = np.linalg.norm( self.X[0:3] - agent.X[0:3] )**2 - beta*d_min**2   
            s = ( self.X[0:3] - agent.X[0:3]).T @ np.array( [np.cos(theta),np.sin(theta),0] ).reshape(-1,1)
            h = h - self.sigma(s)
            
            der_sigma = self.sigma_der(s)
            dh_dxi = np.append( np.append(2*( self.X[0:2] - agent.X[0:2] ).T,[[0]], axis=1) - der_sigma * ( np.array([ [np.cos(theta), np.sin(theta), 0] ]) ),  - der_sigma * ( -np.sin(theta)*( self.X[0,0]-agent.X[0,0] ) + np.cos(theta)*( self.X[1,0] - agent.X[1,0] ) ) , axis=1)
            
            if agent.type=='SingleIntegrator3D':
                dh_dxj = -2*np.append( ( self.X[0:2] - agent.X[0:2] ).T, [[0]], axis=1 ) + der_sigma * ( np.array([ [np.cos(theta), np.sin(theta), 0] ]) )
            elif agent.type=='SingleIntegrator6D':
                dh_dxj = np.append( -2*np.append( ( self.X[0:2] - agent.X[0:2] ).T, [[0]], axis=1 ) + der_sigma * ( np.array([ [np.cos(theta), np.sin(theta), 0] ]) ), [[0,0,0]]  , axis=1 )
            elif agent.type=='Unicycle2D':
                dh_dxj = np.append( -2*np.append( ( self.X[0:2] - agent.X[0:2] ).T, [[0]] , axis=1) + der_sigma * ( np.array([ [np.cos(theta), np.sin(theta), 0] ]) ), np.array([[0]]), axis=1 )
            elif agent.type=='DoubleIntegrator3D':
                dh_dxj = np.append( -2*np.append( ( self.X[0:2] - agent.X[0:2] ).T, [[0]], axis=1 ) + der_sigma * ( np.array([ [np.cos(theta), np.sin(theta), 0] ]) ), [[0,0,0]]  , axis=1 )
            else:
                dh_dxj = -2*np.append( ( self.X[0:2] - agent.X[0:2] ).T, [[0]], axis=1 ) + der_sigma * ( np.array([ [np.cos(theta), np.sin(theta), 0] ]) )
            return h, dh_dxi, dh_dxj
        
        else:
            # it is a surveillance one
            GX = np.copy(agent.X)
            radii = 0
            if self.X[2,0] <= GX[2,0]: # lower height than the 
                radii =  (GX[2,0]-self.X[2,0]) * np.tan(agent.cone_angle)
            GX[2,0] = np.copy(self.X[2,0])
            h = np.linalg.norm( self.X[0:3] - GX[0:3] )**2 - beta*(radii)**2   
            s = ( self.X[0:3] - GX[0:3]).T @ np.array( [np.cos(theta),np.sin(theta),0] ).reshape(-1,1)
            h = h - self.sigma(s)
            
            der_sigma = self.sigma_der(s)
            dh_dxi = np.append( np.append(2*( self.X[0:2] - GX[0:2] ).T,[[0]], axis=1) - der_sigma * ( np.array([ [np.cos(theta), np.sin(theta), 0] ]) ),  - der_sigma * ( -np.sin(theta)*( self.X[0,0]-GX[0,0] ) + np.cos(theta)*( self.X[1,0] - GX[1,0] ) ) , axis=1)
            dh_dxj = np.append( -2*np.append( ( self.X[0:2] - GX[0:2] ).T, [[0]], axis=1 ) + der_sigma * ( np.array([ [np.cos(theta), np.sin(theta), 0] ]) ), [[0,0,0]]  , axis=1 )
            return h, dh_dxi, dh_dxj
            