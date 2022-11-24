import numpy as np
import matplotlib.pyplot as plt

class SingleIntegrator3D:
    
    def __init__(self,X0,dt,ax,id, color='r',palpha=1.0, plot=True):
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

        self.U = np.array([0,0,0]).reshape(-1,1)
       
        # Plot handles
        self.plot = plot
        if self.plot:
            self.body = ax.scatter([],[],[],c=color,alpha=palpha,s=10)
            self.render_plot()
        self.Xs = np.copy(self.X)
        self.Us = np.copy(self.U)
        
    def f(self):
        return np.array([0,0,0]).reshape(-1,1)
    
    def g(self):
        return np.eye(3)
        
    def step(self,U): #Just holonomic X,T acceleration

        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        self.render_plot()
        self.Xs = np.append(self.Xs,self.X,axis=1)
        self.Us = np.append(self.Us,self.U,axis=1)
        return self.X
    
    def Xdot(self):
        return self.f() + self.g() @ self.U

    def render_plot(self):
        if self.plot:
            x = np.array([self.X[0,0],self.X[1,0],self.X[2,0]])

            # scatter plot update
            self.body._offsets3d = ([[x[0]],[x[1]],[x[2]]])
            
    def lyapunov(self, G, type='none'):
        V = np.linalg.norm( self.X[0:3] - G[0:3] )**2
        dV_dxi = 2*( self.X[0:3] - G[0:3] ).T
        
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
    
    def agent_barrier(self, agent, d_min):
        
        h = np.linalg.norm( self.X - agent.X[0:3] )**2 - d_min**2
        dh_dxi = 2*( self.X - agent.X[0:3] ).T
        
        if agent.type=='SingleIntegrator3D':
            dh_dxj = -2*( self.X - agent.X[0:3] ).T
        elif agent.type=='SingleIntegrator6D':
            dh_dxj = np.append( -2*( self.X - agent.X[0:3] ).T, [[0, 0, 0]], axis = 1 )
        elif agent.type=='Unicycle2D':
            dh_dxj = np.append( -2*( self.X - agent.X[0:3] ).T, [[0]], axis=1  )
        elif agent.type=='DoubleIntegrator3D':
            dh_dxj = np.append( -2*( self.X - agent.X[0:3] ).T, [[0, 0, 0]], axis = 1 )
        else:
            dh_dxj = -2*( self.X - agent.X[0:3] ).T
            
# For testing only!    
if 0:           
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection ="3d")#,xlim=(-1,5),ylim=(-0.5,5))             
    robot = SingleIntegrator3D(np.array([0,0,0]), 0.05, ax, 0)

    plt.ion()
    for i in range(100):
        robot.step(np.array([0.2,0.2,0.2]))
        fig.canvas.draw()
        fig.canvas.flush_events()