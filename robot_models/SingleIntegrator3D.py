import numpy as np
import matplotlib.pyplot as plt
from trust_utils import *

class SingleIntegrator3D:
    
    def __init__(self,X0,dt,ax,id = 0, mode = 'ego', grounded = False, target = 0, color='r',alpha = 0.8, palpha=1.0, plot=True, nominal_plot=True, num_constraints = 0, num_robots = 1):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'SingleIntegrator3D'        
        
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
            
        
        # to store constraints
        self.A = np.zeros((num_constraints,3))
        self.b = np.zeros((num_constraints,1))
        self.agent_objective = [0] * num_robots
        self.U_ref = np.array([0,0,0]).reshape(-1,1)
        self.U_nominal = np.array([0,0,0]).reshape(-1,1)
        self.alpha = alpha*np.ones((1,num_robots))
        
        # trust
        self.trust = np.zeros((1,num_robots))
        self.h = np.zeros((1,num_robots))
        
        # plot
        self.Xs = np.copy(self.X)
        self.Us = np.copy(self.U)
        self.alphas = np.copy(self.alpha)
        self.trusts = np.copy(self.trust)
        self.hs = np.copy(self.h)
        
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
        self.hs = np.append(self.hs,self.h,axis=0)
        self.trusts = np.append(self.trusts, self.trust, axis=0)
        self.alphas = np.append(self.alphas, self.alpha, axis=0)
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
            x = np.array([self.X_nominal[0,0],self.X_nominal[1,0],self.X_nominal[2,0]])
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
    
    def lyapunov_nominal(self, G, type='none'):
        V = np.linalg.norm( self.X_nominal[0:3] - G[0:3] )**2
        dV_dxi = 2*( self.X_nominal[0:3] - G[0:3] ).T
        
        if type=='SingleIntegrator3D':
            dV_dxj = -2*( self.X_nominal[0:3] - G[0:3] ).T
        elif type=='SingleIntegrator6D':
            dV_dxj = np.append( -2*( self.X_nominal[0:3] - G[0:3] ).T, [[0, 0, 0]], axis = 1 )
        elif type=='Unicycle2D':
            dV_dxj = np.append( -2*( self.X_nominal[0:3] - G[0:3] ).T, [[0]], axis=1  )
        elif type=='DoubleIntegrator3D':
            dV_dxj = np.append( -2*( self.X_nominal[0:3] - G[0:3] ).T, [[0, 0, 0]], axis = 1 )
        else:
            dV_dxj = -2*( self.X_nominal[0:3] - G[0:3] ).T
        
        return V, dV_dxi, dV_dxj
    
    def nominal_input( self, G, type='none' ):
        V, dV_dxi, dV_dxj = self.lyapunov( G, type )
        return -3.0*dV_dxi.T/np.linalg.norm(dV_dxi)
    
    def agent_barrier(self, agent, d_min):
        
        if agent.type != 'Surveillance':
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
            radii = 0
            if self.X[2,0] < GX[2,0]: # lower height than the 
                radii =  (GX[2,0]-self.X[2,0]) * np.tan(agent.cone_angle)
            GX[2,0] = self.X[2,0]
            h = np.linalg.norm( self.X[0:3] - GX[0:3] )**2 - (d_min + radii)**2
            dh_dxi = 2*( self.X[0:3] - GX[0:3] ).T
        
            dh_dxj = np.append( -2*( self.X[0:3] - GX[0:3] ).T, [[0, 0, 0]], axis = 1 )
        
        return h, dh_dxi, dh_dxj    
    
    def trust_param_update( self, agent, id, d_min, uT, min_dist, h_min, alpha_der_max, dt = 0.01 ):
        h, dh_dxi, dh_dxj = self.agent_barrier(agent, d_min)  
        A = dh_dxj #@ robots[j].g()
        b = -self.alpha[0,id] * h  - dh_dxi @ ( self.f() + self.g() @ uT ) #- dh_dxj @ robots[j].f() #- dh_dxi @ robots[j].U                    
        self.trust[0,id], asserted = compute_trust( A, b, agent.f() + agent.g() @ agent.U, agent.x_dot_nominal, h, min_dist, h_min )  
        self.alpha[0,id] = self.alpha[0,id] + alpha_der_max * self.trust[0,id]
    
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
        
        