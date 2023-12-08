import numpy as np

class DoubleIntegrator2D:
    
    def __init__(self,X0,dt,ax,id = 0, color='r',palpha=1.0, plot=True):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'SingleIntegrator2D'        
        
        X0 = X0.reshape(-1,1)
        self.X = X0
        self.dt = dt
        self.id = id
        self.color = color
        self.palpha = palpha

        self.U = np.array([0,0]).reshape(-1,1)
        self.xdot = np.zeros((2,1))
       
        # Plot handles
        self.plot = plot
        if self.plot:
            self.ax = ax
            self.body = ax.scatter([],[],c=color,alpha=palpha,s=30)
            self.render_plot()
        self.Xs = np.copy(self.X)
        self.Us = np.copy(self.U)
        
    def f(self):
        return np.array([self.X[2,0],self.X[3,0],0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [0, 0], [0, 0], [1, 0],[0, 1] ])
        
    def step(self,U): #Just holonomic X,T acceleration

        self.U = U.reshape(-1,1)
        self.xdot = self.f() + self.g() @ self.U
        self.X = self.X + self.xdot * self.dt        
        self.render_plot()
        self.Xs = np.append(self.Xs,self.X,axis=1)
        self.Us = np.append(self.Us,self.U,axis=1)
        return self.X

    def render_plot(self):
        if self.plot:
            x = np.array([self.X[0,0],self.X[1,0]])
            self.body.set_offsets([x[0],x[1]])
            
    def lyapunov(self, G):
        V = np.linalg.norm( self.X - G )**2
        dV_dxi = 2*( self.X - G ).T
        dV_dxj = -2*( self.X - G ).T
        return V, dV_dxi, dV_dxj
    
    def obstacle_barrier(self, agent, d_min, ):
        h = np.linalg.norm( self.X[0:2] - agent.X[0:2] )**2 - d_min**2\
        
        # assert(h>=-0.02)
        # if (h<=0):
        #     h=0.01
        h_dot = 2 * (self.X[0:2]-agent.X[0:2]).T @ (self.X[2:4]-agent.U)
        dh_dot_dxi = 2 * np.append( (self.X[2:4]-agent.U).T, (self.X[0:2]-agent.X[0:2]).T, axis=1 )
        dh_dot_dxj = -2 * (self.X[2:4]-agent.U).T
        return h, h_dot, dh_dot_dxi, dh_dot_dxj