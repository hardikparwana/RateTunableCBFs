import numpy as np
from utils.utils import wrap_angle
from trust_utils import *

class SI:
    
    def __init__(self,X0,dt,ax,id = 0, mode = 'ego', target = 0, color='r',alpha = 0.8, palpha=1.0,plot=True, nominal_plot=True, num_constraints = 0, num_robots = 1):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'SI2D'
        
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
            self.body = ax.scatter([],[],alpha=palpha,s=60,facecolors=self.color,edgecolors=self.color) #facecolors='none'
            self.render_plot()
        if self.plot_nominal:
            self.body_nominal = ax.scatter([],[],alpha=0.3,s=60,facecolors=self.color,edgecolors=self.color) #facecolors='none'
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
        return np.array([0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [ 1, 0],
                          [0, 1] ])  
        
    def f_nominal(self):
        return np.array([0,0]).reshape(-1,1)
    
    def g_nominal(self):
        return np.array([ [ 1, 0 ],
                          [0, 1] ])  
        
    def Xdot(self):
        return self.f() + self.g() @ self.U     
         
    def step(self,U): 
        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        self.render_plot()
        self.Xs = np.append(self.Xs,self.X,axis=1)
        self.Us = np.append(self.Us,self.U,axis=1)
        return self.X
    
    def step_nominal(self,U): 
        self.U_nominal = U.reshape(-1,1)
        self.X_nominal = self.X_nominal + ( self.f_nominal() + self.g_nominal() @ self.U_nominal )*self.dt
        self.render_plot_nominal()
        return self.X_nominal
    
    def render_plot(self):
        if self.plot:
            x = np.array([self.X[0,0],self.X[1,0]])
            self.body.set_offsets([x[0],x[1]])
    
    def render_plot_nominal(self):
        if self.plot_nominal:
            x = np.array([self.X_nominal[0,0],self.X_nominal[1,0]])
            self.body_nominal.set_offsets([x[0],x[1]])
    
    def lyapunov(self, G, type='none'):
        V = np.linalg.norm( self.X[0:2] - G[0:2] )**2
        dV_dxi = np.append( 2*( self.X[0:2] - G[0:2] ).T, [[0]] , axis = 1 )
        dV_dxj = -2*( self.X[0:2] - G[0:2] ).T
        
        return V, dV_dxi, dV_dxj
    