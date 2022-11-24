import numpy as np

class SingleIntegrator2D:
    
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
       
        # Plot handles
        self.plot = plot
        if self.plot:
            self.ax = ax
            self.body = ax.scatter([],[],[],c=color,alpha=palpha,s=30)
            self.render_plot()
        self.Xs = np.copy(self.X)
        self.Us = np.copy(self.U)
        
    def f(self):
        return np.array([0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [1, 0],[0, 1] ])
        
    def step(self,U): #Just holonomic X,T acceleration

        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        self.render_plot()
        self.Xs = np.append(self.Xs,self.X,axis=1)
        self.Us = np.append(self.Us,self.U,axis=1)
        return self.X

    def render_plot(self):
        if self.plot:
            x = np.array([self.X[0,0],self.X[1,0],0])
            self.body._offsets3d = ([[x[0]],[x[1]],[x[2]]])
            
    def lyapunov(self, G):
        V = np.linalg.norm( self.X - G )**2
        dV_dxi = 2*( self.X - G ).T
        dV_dxj = -2*( self.X - G ).T
        return V, dV_dxi, dV_dxj