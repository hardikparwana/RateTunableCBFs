import numpy as np
from utils.utils import wrap_angle
from trust_utils import *
import matplotlib.patches as mpatches

class Unicycle:
    
    def __init__(self,X0,dt,ax,id = 0, mode = 'ego', target = 0, color='r',alpha = 0.8, palpha=1.0,plot=True, nominal_plot=True, num_constraints = 0, num_robots = 1, plot_fov=True):
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
        self.FoV_angle = np.pi/3
        self.FoV_length = self.max_D
        
        
        self.U = np.array([0,0]).reshape(-1,1)
        
        # Plot handles
        self.plot = plot
        self.plot_nominal = nominal_plot
        self.plot_fov = plot_fov
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
            
        if self.plot_fov:     
            self.lines, = ax.plot([],[],'o-')
            self.poly = mpatches.Polygon([(0,0.2)], closed=True, color='r',alpha=0.1, linewidth=0) #[] is Nx2
            self.fov_arc = ax.add_patch(self.poly)
            self.areas, = ax.fill([],[],'r',alpha=0.1)
            self.body = ax.scatter([],[],c=color,s=10)            
            self.des_point = ax.scatter([],[],s=10, facecolors='none', edgecolors='r')
            
            self.render_plot_fov()
            
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
        return np.array([0,0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [ np.cos(self.X[2,0]), 0],
                          [ np.sin(self.X[2,0]), 0],
                          [0, 1] ])  
        
    def f_nominal(self):
        return np.array([0,0,0]).reshape(-1,1)
    
    def g_nominal(self):
        return np.array([ [ np.cos(self.X_nominal[2,0]), 0 ],
                          [ np.sin(self.X_nominal[2,0]), 0],
                          [0, 1] ])  
        
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
    
    def render_plot_fov(self): #,lines,areas,body, poly, des_point):
        # length = 3
        # FoV = np.pi/3   # 60 degrees

        x = np.array([self.X[0,0],self.X[1,0]])
  
        theta = self.X[2][0]
        theta1 = theta + self.FoV_angle/2
        theta2 = theta - self.FoV_angle/2
        e1 = np.array([np.cos(theta1),np.sin(theta1)])
        e2 = np.array([np.cos(theta2),np.sin(theta2)])

        P1 = x + self.FoV_length*e1
        P2 = x + self.FoV_length*e2  

        des_dist = self.min_D + (self.max_D - self.min_D)/2
        des_x = np.array( [ self.X[0,0] + np.cos(theta)*des_dist, self.X[1,0] + np.sin(theta)*des_dist    ] )

        triangle_hx = [x[0] , P1[0], P2[0], x[0] ]
        triangle_hy = [x[1] , P1[1], P2[1], x[1] ]
        
        triangle_v = [ x,P1,P2,x ]  

        # lines.set_data(triangle_hx,triangle_hy)
        self.areas.set_xy(triangle_v)

        # scatter plot update
        self.body.set_offsets([x[0],x[1]])
        self.des_point.set_offsets([des_x[0], des_x[1]])

        #Fov arc
        self.poly.set_xy(self.arc_points(x, self.FoV_length, theta2, theta1))
        
    def arc_points(self, center, radius, theta1, theta2, resolution=50):
        # generate the points
        theta = np.linspace(theta1, theta2, resolution)
        points = np.vstack((radius*np.cos(theta) + center[0], 
                            radius*np.sin(theta) + center[1]))
        return points.T

    
    def lyapunov(self, G, type='none'):
        V = np.linalg.norm( self.X[0:2] - G[0:2] )**2
        dV_dxi = np.append( 2*( self.X[0:2] - G[0:2] ).T, [[0]] , axis = 1 )
        dV_dxj = -2*( self.X[0:2] - G[0:2] ).T
        
        return V, dV_dxi, dV_dxj
    
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
    
    def sigma(self,s):
        k1 = 1.0
        k2 = 1.0
        return k2 * (np.exp(k1-s)-1)/(np.exp(k1-s)+1)
    
    def sigma_der(self,s):
        k1 = 1.0
        k2 = 1.0    
        return - k2 * np.exp(k1-s)/( 1+np.exp( k1-s ) ) * ( 1 - self.sigma(s)/k2 )
    
    def agent_barrier(self, agent, d_min):
        
        beta = 1.01
        theta = self.X[2,0]
        
        if agent.type!='Surveillance':
        
            h = np.linalg.norm( self.X[0:2] - agent.X[0:2] )**2 - beta*d_min**2   
            s = ( self.X[0:2] - agent.X[0:2]).T @ np.array( [np.cos(theta),np.sin(theta),0] ).reshape(-1,1)
            h = h - self.sigma(s)
            
            der_sigma = self.sigma_der(s)
            dh_dxi = np.append( 2*( self.X[0:2] - agent.X[0:2] ).T - der_sigma * np.array([ [np.cos(theta), np.sin(theta)] ]),  - der_sigma * ( -np.sin(theta)*( self.X[0,0]-agent.X[0,0] ) + np.cos(theta)*( self.X[1,0] - agent.X[1,0] ) ) , axis=1)
            dh_dxj = -2*( self.X[0:2] - agent.X[0:2] ).T + der_sigma * np.array([ [np.cos(theta), np.sin(theta), 0] ])
            return h, dh_dxi, dh_dxj
        
    def trust_param_update( self, agent, id, d_min, uT, min_dist, h_min, alpha_der_max, dt = 0.01 ):
        h, dh_dxi, dh_dxj = self.agent_barrier(agent, d_min)  
        A = dh_dxj #@ robots[j].g()
        b = -self.alpha[0,id] * h  - dh_dxi @ ( self.f() + self.g() @ uT ) #- dh_dxj @ robots[j].f() #- dh_dxi @ robots[j].U                    
        self.trust[0,id], asserted = compute_trust( A, b, agent.f() + agent.g() @ agent.U, agent.x_dot_nominal, h, min_dist, h_min )  
        self.alpha[0,id] = self.alpha[0,id] + alpha_der_max * self.trust[0,id]
            