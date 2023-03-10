import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
from scipy.spatial.transform import Rotation as R

class Surveillance:
    
    def __init__(self,X0,dt,ax,id = 0, mode = 'ego', target = 0, color='r',palpha=1.0, plot=True, nominal_plot=True, cone_length = 3, cone_angle = np.pi/3, num_robots = 1, num_constraints = 0):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'Surveillance'        
        
        X0 = X0.reshape(-1,1)
        self.X = X0
        self.X_nominal = np.copy(self.X)
        
        self.target = target
        self.mode = mode
        self.dt = dt
        self.id = id
        self.color = color
        self.palpha = palpha

        self.U = np.array([0,0,0,0,0,0]).reshape(-1,1)
       
        self.ax = ax
        self.cone_resolution = 30
        self.cone_length = cone_length
        height = cone_length*np.cos(cone_angle)
        self.cone_angle = cone_angle
        r = np.linspace( 0, cone_length*np.sin(cone_angle), self.cone_resolution )  # radius changes from 0 to Max value
        h = np.linspace( 0, height, self.cone_resolution )
        th = np.linspace( 0, 2*np.pi, self.cone_resolution )
        R, T = np.meshgrid( r, th )
        H, T = np.meshgrid( h, th )
        self.coneX = R * np.cos(T)
        self.coneY = R * np.sin(T)
        self.coneZ = -H
        self.coneP = np.append( self.coneX.reshape(1,-1), self.coneY.reshape(1,-1), axis=0 )
        self.coneP = np.append( self.coneP, self.coneZ.reshape(1,-1), axis=0 )
        
        # Plot handles
        self.plot = plot
        self.plot_nominal = nominal_plot
        if self.plot:
            self.body = ax.scatter([],[],[],c=self.color,alpha=palpha,s=40)
            self.cone = ax.plot_surface(self.coneX, self.coneY, self.coneZ, alpha=0.4,linewidth=0, color = self.color)#, cmap=cm.coolwarm)
            self.render_plot()
        if self.plot_nominal:
            self.body_nominal = ax.scatter([],[],[],c=self.color,alpha=palpha,s=40)
            self.render_plot_nominal()
            
        self.Xs = np.copy(self.X)
        self.Us = np.copy(self.U)
        
        self.U_ref = np.array([0,0]).reshape(-1,1)
        self.U_nominal = np.array([0,0]).reshape(-1,1)
        
    def f(self):
        return np.array([0,0,0,0,0,0]).reshape(-1,1)
    
    def g(self):
        return np.eye(6)
        
    def step(self,U): #Just holonomic X,T acceleration

        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        # self.render_plot()
        self.Xs = np.append(self.Xs,self.X,axis=1)
        self.Us = np.append(self.Us,self.U,axis=1)
        return self.X
    
    def step_nominal(self,U): #Just holonomic X,T acceleration
        self.U_nominal = U.reshape(-1,1)
        self.X_nominal = self.X_nominal + ( self.f() + self.g() @ self.U_nominal )*self.dt
        # self.render_plot_nominal()
        return self.X_nominal
    
    def Xdot(self):
        return self.f() + self.g() @ self.U

    def render_plot(self):
        if self.plot:
            x = np.array([self.X[0,0],self.X[1,0],self.X[2,0]])

            # scatter plot update
            self.body._offsets3d = ([[x[0]],[x[1]],[x[2]]])
            
            # rot_mat = np.array( [ [ np.cos(self.X[5,0]), np.sin(self.X[5,0]), 0 ],
            #                       [ -np.sin(self.X[5,0]), np.cos(self.X[5,0]), 0 ],
            #                       [ 0, 0, 1]] )
            rot_mat = R.from_euler( 'zyx', [ self.X[5,0], self.X[4,0], self.X[3,0] ]  )
            
            trans = np.copy(self.X[0:3])
            points = trans + rot_mat.as_matrix() @ self.coneP
            xmesh = points[0,:].reshape((self.cone_resolution,self.cone_resolution))
            ymesh = points[1,:].reshape((self.cone_resolution,self.cone_resolution))
            zmesh = points[2,:].reshape((self.cone_resolution,self.cone_resolution))
            
            self.cone.remove()
            self.cone = self.ax.plot_surface( xmesh, ymesh, zmesh, alpha=0.4,linewidth=0, color=self.color )
            
    def render_plot_nominal(self):
        if self.plot_nominal:
            x = np.array([self.X[0,0],self.X[1,0],self.X[2,0]])
            self.body_nominal._offsets3d = ([[x[0]],[x[1]],[x[2]]])
            
    def lyapunov(self, G, type='none'):
        V = np.linalg.norm( self.X[0:3] - G[0:3] )**2
        dV_dxi = np.append( 2*( self.X[0:3] - G[0:3] ).T, [[0, 0, 0]] , axis = 1 )
        
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
    
    def nominal_input(self, G, type='none'):
        V, dV_dxi, dV_dxj = self.lyapunov(self, G, type)
        return -3.0*dV_dxi.T/np.linalg.norm(dV_dxi)
    
    def agent_barrier(self, agent, d_min):
        
        h = np.linalg.norm( self.X[0:3] - agent.X[0:3] )**2 - d_min**2
        dh_dxi = np.append( 2*( self.X[0:3] - agent.X[0:2] ).T, [[0, 0, 0]]  , axis=1  )
        
        if agent.type=='SingleIntegrator3D':
            dh_dxj = -2*( self.X[0:3] - agent.X[0:2] ).T
        elif agent.type=='SingleIntegrator6D':
            dh_dxj = np.append( -2*( self.X[0:3] - agent.X[0:2] ).T, [[0, 0, 0]], axis = 1 )
        elif agent.type=='Unicycle2D':
            dh_dxj = np.append( -2*( self.X[0:3] - agent.X[0:2] ).T, [[0]], axis=1  )
        elif agent.type=='DoubleIntegrator3D':
            dh_dxj = np.append( -2*( self.X[0:3] - agent.X[0:2] ).T, [[0, 0, 0]], axis = 1 )
        else:
            dh_dxj = -2*( self.X[0:3] - agent.X[0:3] ).T
            

# For testing only!            
if 0:  
    # Actually not sure about the math here though:
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)

        
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection ="3d",xlim=(-1,5),ylim=(-1,5), zlim=(-1,1))   
    # ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.zaxis.pane.set_edgecolor('r')
    x = [-5,-5,5,5]
    y = [-5,5,5,-5]
    z = [0,0,0,0]
    verts = [list(zip(x,y,z))]
    ax.add_collection3d(Poly3DCollection(verts,facecolor='gray', alpha=0.5))       
    robot = Surveillance(np.array([0,0,2,0,0,0]), 0.05, ax, 0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.ion()
    for i in range(100):
        robot.step(np.array([0.5,0.0,0.0,0.0,0,0]))
        fig.canvas.draw()
        fig.canvas.flush_events()