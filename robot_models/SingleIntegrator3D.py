import numpy as np
import matplotlib.pyplot as plt

class SingleIntegrator3D:
    
    def __init__(self,X0,dt,ax,id = 0, mode = 'ego', target = 0, color='r',palpha=1.0, plot=True, nominal_plot=True):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'SingleIntegrator2D'        
        
        X0 = X0.reshape(-1,1)
        self.X = X0
        
        self.X_nominal = np.copy(self.X)
        self.target = target
        
        self.dt = dt
        self.id = id
        self.color = color
        self.palpha = palpha

        self.U = np.array([0,0,0]).reshape(-1,1)
       
        # Plot handles
        self.plot = plot
        self.plot_nominal = nominal_plot
        if self.plot:
            self.body = ax.scatter([],[],[],c=color,alpha=palpha,s=10)
            self.render_plot()
        if self.plot_nominal:
            self.body_nominal = ax.scatter([],[],[],c=color,alpha=0.3,s=10)
            self.render_plot_nominal()
            
        self.Xs = np.copy(self.X)
        self.Us = np.copy(self.U)
        
    def f(self):
        return np.array([0,0,0]).reshape(-1,1)
    
    def g(self):
        if self.grounded:
            G = np.eye(3)
            G[2,2] = 0;
            return G
        else:
            return np.eye(3)
        
    def step(self,U): #Just holonomic X,T acceleration

        # check for ground condition
        
        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        self.render_plot()
        self.Xs = np.append(self.Xs,self.X,axis=1)
        self.Us = np.append(self.Us,self.U,axis=1)
        return self.X
    
    def step_nominal(self,U): #Just holonomic X,T acceleration
        self.U_nominal = U.reshape(-1,1)
        self.X_nominal = self.X_nominal + ( self.f() + self.g() @ self.U_nominal )*self.dt
        self.render_plot_nominal()
        return self.X_nominal
    
    def Xdot(self):
        return self.f() + self.g() @ self.U

    def render_plot(self):
        if self.plot:
            x = np.array([self.X[0,0],self.X[1,0],self.X[2,0]])

            # scatter plot update
            self.body._offsets3d = ([[x[0]],[x[1]],[x[2]]])
            
    def render_plot_nominal(self):
        if self.plot_nominal:
            x = np.array([self.X[0,0],self.X[1,0],self.X[2,0]])
            self.body_nominal._offsets3d = ([[x[0]],[x[1]],[x[2]]])
            
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
    
    def nominal_input( self, G, type='none' ):
        V, dV_dxi, dV_dxj = self.lyapunov( G, type )
        return -3.0*dV_dxi.T/np.linalg.norm(dV_dxi)
    
    def agent_barrier(self, agent, d_min):
        
        if agent.type != 'SingleIntegrator6D':
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
        else:
            # it is a surveillance one
            GX = np.copy(agent.X)
            if self.X[2,0] < GX[2,0]: # lower height than the 
                
            h = np.linalg.norm( self.X - agent.X[0:3] )**2 - d_min**2
            dh_dxi = 2*( self.X - agent.X[0:3] ).T
        
            dh_dxj = np.append( -2*( self.X - agent.X[0:3] ).T, [[0, 0, 0]], axis = 1 )
        
        return h, dh_dxi, dh_dxj    
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
        
        