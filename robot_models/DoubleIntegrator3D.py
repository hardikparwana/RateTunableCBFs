import numpy as np
import matplotlib.pyplot as plt

class DoubleIntegrator3D:
    
    def __init__(self,X0,dt,ax,id = 0, mode = 'ego', grounded = False, target = 0, color='r', alpha = 0.8, palpha=1.0, plot=True, nominal_plot = True, num_constraints = 0, num_robots = 1):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'DoubleIntegrator3D'        
        
        X0 = X0.reshape(-1,1)
        self.X = X0
        self.X_nominal = np.copy(self.X)
        
        self.target = target
        self.mode = mode
        self.grounded = grounded
        
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
        
        # to store constraints
        self.A = np.zeros((num_constraints,3))
        self.b = np.zeros((num_constraints,1))
        self.agent_objective = [0] * num_robots
        self.U_ref = np.array([0,0,0]).reshape(-1,1)
        self.U_nominal = np.array([0,0,0]).reshape(-1,1)
        self.alpha = alpha * np.ones((1,num_robots))
        self.alpha1 = alpha/2 * np.ones((1,num_robots))
        
    def f(self):
        fm = np.zeros((6,1))
        fm[0,0] = self.X[3,0]; fm[1,0] = self.X[4,0]; fm[2,0] = self.X[5,0];
        return fm
    
    def g(self):
        if self.grounded:
            G = np.append( np.zeros((3,3)), np.eye(3), axis=0 )
            G[5,5] = 0.0
            return G
        else:
            return np.append( np.zeros((3,3)), np.eye(3), axis=0 )
        
    def f_nominal(self):
        fm = np.zeros((6,1))
        fm[0,0] = self.X_nominal[3,0]; fm[1,0] = self.X_nominal[4,0]; fm[2,0] = self.X_nominal[5,0];
        return fm
    
    def g_nominal(self):
        if self.grounded:
            G = np.append( np.zeros((3,3)), np.eye(3), axis=0 )
            G[5,5] = 0.0
            return G
        else:
            return np.append( np.zeros((3,3)), np.eye(3), axis=0 )
        
    def step(self,U): #Just holonomic X,T acceleration

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
    
    def Xdot(self):
        return self.f() + self.g() @ self.U

    def render_plot(self):
        if self.plot:
            x = np.array([self.X[0,0],self.X[1,0],self.X[2,0]])

            # scatter plot update
            self.body._offsets3d = ([[x[0]],[x[1]],[x[2]]])
            
    def render_plot_nominal(self):
        if self.plot_nominal:
            x = np.array([self.X_nominal[0,0],self.X_nominal[1,0],self.X_nominal[2,0]])
            self.body_nominal._offsets3d = ([[x[0]],[x[1]],[x[2]]])
            
    def lyapunov(self, G, type='none'): # position based only: will not work for anything lol
        k1 = 3
        V = k1 * np.linalg.norm( self.X[0:3] - G[0:3] )**2 + np.linalg.norm( self.X[3:6] )**2
        dV_dxi = np.append( 2*k1*( self.X[0:3] - G[0:3] ).T, 2*self.X[3:6].T  , axis=1  )
        
        if type=='SingleIntegrator3D':
            dV_dxj = -2*( self.X[0:3] - G[0:3] ).T
        elif type=='SingleIntegrator6D':
            dV_dxj = np.append( -2*( self.X[0:3] - G[0:3] ).T, [[0, 0, 0]], axis = 1 )
        elif type=='Unicycle2D':
            dV_dxj = np.append( -2*( self.X[0:3] - G[0:3] ).T, [[0]], axis=1  )
        elif type=='DoubleIntegrator3D':
            dV_dxj = np.append( -2*( self.X[0:3] - G[0:3] ).T, [[0,0,0]], axis=1 )
        else:
            dV_dxj = -2*( self.X[0:3] - G[0:3] ).T
        
        return V, dV_dxi, dV_dxj
    
    def lyapunov_nominal(self, G, type='none'): # position based only: will not work for anything lol
        k1 = 3
        V = k1 * np.linalg.norm( self.X_nominal[0:3] - G[0:3] )**2 + np.linalg.norm( self.X_nominal[3:6] )**2
        dV_dxi = np.append( 2*( self.X_nominal[0:3] - G[0:3] ).T, 2*self.X_nominal[3:6].T  , axis=1  )
        
        if type=='SingleIntegrator3D':
            dV_dxj = -2*( self.X_nominal[0:3] - G[0:3] ).T
        elif type=='SingleIntegrator6D':
            dV_dxj = np.append( -2*( self.X_nominal[0:3] - G[0:3] ).T, [[0, 0, 0]], axis = 1 )
        elif type=='Unicycle2D':
            dV_dxj = np.append( -2*( self.X_nominal[0:3] - G[0:3] ).T, [[0]], axis=1  )
        elif type=='DoubleIntegrator3D':
            dV_dxj = np.append( -2*( self.X_nominal[0:3] - G[0:3] ).T, [[0,0,0]], axis=1 )
        else:
            dV_dxj = -2*( self.X_nominal[0:3] - G[0:3] ).T
        
        return V, dV_dxi, dV_dxj
    
    def nominal_input( self, G, type='none' ):
        # V, dV_dxi, dV_dxj = self.lyapunov( G, type )
        # if np.linalg.norm( dV_dxi[0,3:6].reshape(1,-1) ) > 0.01:
        #     return - 3.0 * np.abs(dV_dxi[0,0:3].reshape(1,-1) @ self.Xdot()[0:3] )* dV_dxi[0,3:6].reshape(-1,1) / np.linalg.norm( dV_dxi[0,3:6].reshape(1,-1) )
        # else:
        #     return np.array([0,0,0]).reshape(-1,1)
        
        vd = 2.0 * ( G[0:3] - self.X[0:3] )
        ad = 4.0 * ( vd - self.X[3:6] )
        return ad
    
    def agent_barrier(self, agent, d_min): # assuming velocity available??
                
        if agent.type!='Surveillance':
            h1 = np.linalg.norm( self.X[0:3] - agent.X[0:3] )**2 - d_min**2        
            h1_dot = 2*( self.X[0:3] - agent.X[0:3] ).T @ ( self.X[3:6] - agent.Xdot()[0:3] )
            
            h2 = h1_dot + self.alpha1[0,agent.id] * h1
            
            assert(h2>=0)
            
            dh2_dxi = np.append( 2*(self.X[3:6] - agent.Xdot()[0:3]).T + 2*self.alpha1[0,agent.id]*( self.X[0:3]-agent.X[0:3] ).T,  2*(self.X[0:3] - agent.X[0:3]).T  , axis=1)
            
            if agent.type=='SingleIntegrator3D':  # ?? = 0??
                dh2_dxj = -2*(self.X[3:6] - agent.Xdot()[0:3]).T - 2*self.alpha1[0,agent.id]*( self.X[0:3]-agent.X[0:3] ).T
            elif agent.type=='SingleIntegrator6D':
                dh2_dxj = np.append( -2*( self.X[3:6] - agent.Xdot()[0:3] ).T - 2*self.alpha1[0,agent.id]*( self.X[0:3]-agent.X[0:3] ).T , [[0, 0, 0]], axis = 1 )
            elif agent.type=='Unicycle2D':
                dh2_dxj = np.append( -2*( self.X[3:6] - agent.Xdot()[0:3] ).T - 2*self.alpha1[0,agent.id]*( self.X[0:3]-agent.X[0:3] ).T, [[0]], axis=1  )
            elif agent.type=='DoubleIntegrator3D':
                dh2_dxj = np.append( -2*(self.X[3:6] - agent.Xdot()[0:3]).T - 2*self.alpha1[0,agent.id]*( self.X[0:3]-agent.X[0:3] ).T,  -2*(self.X[0:3] - agent.X[0:3]).T  , axis=1)
            else:
                dh2_dxj = -2*(self.X[3:6] - agent.Xdot()[0:3]).T - 2*self.alpha1[0,agent.id]*( self.X[0:3]-agent.X[0:3] ).T         
            return h2, dh2_dxi, dh2_dxj
        else:
             # it is a surveillance one
            GX = np.copy(agent.X)
            radii = 0
            if self.X[2,0] < GX[2,0]: # lower height than the 
                radii =  (GX[2,0]-self.X[2,0]) * np.tan(agent.cone_angle)
            GX[2,0] = self.X[2,0]
            h1 = np.linalg.norm( self.X[0:3] - GX[0:3] )**2 - (d_min + radii)**2
            h1_dot = 2*( self.X[0:3] - agent.X[0:3] ).T @ ( self.X[3:6] - agent.Xdot()[0:3] )
            h2 = h1_dot + self.alpha1[0,agent.id] * h1
            assert(h2>=0)
            
            dh2_dxi = np.append( 2*(self.X[3:6] - agent.Xdot()[0:3]).T + 2*self.alpha1[0,agent.id]*( self.X[0:3]-agent.X[0:3] ).T,  2*(self.X[0:3] - agent.X[0:3]).T  , axis=1)
            dh2_dxj = np.append( -2*( self.Xdot()[0:3] - agent.Xdot()[0:3] ).T - 2*self.alpha1[0,agent.id]*( self.X[0:3]-agent.X[0:3] ).T, [[0, 0, 0]], axis = 1 )
            return h2, dh2_dxi, dh2_dxj
            
# For testing only!    
if 0:           
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection ="3d")#,xlim=(-1,5),ylim=(-0.5,5))             
    robot = DoubleIntegrator3D(np.array([0,0,0,0,0,0]), 0.05, ax, 0)

    plt.ion()
    for i in range(100):
        robot.step(np.array([0.2,0.2,0.2]))
        fig.canvas.draw()
        fig.canvas.flush_events()