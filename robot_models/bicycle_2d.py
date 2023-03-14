import numpy as np
from utils.utils import wrap_angle
from trust_utils import *
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cvxpy as cp
from robot_models.obstacles import circle2D

class Bicycle_2d:
    
    def __init__(self,X0,dt,ax,id = 0, mode = 'ego', target = 0, color='r',alpha = 0.8, palpha=1.0,plot=True, nominal_plot=True, num_constraints = 0, num_robots = 1, alpha1 = 0.8):
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
        
        self.alpha1 = alpha1
        
        self.U = np.array([0,0]).reshape(-1,1)
        
        # Plot handles
        self.plot = plot
        self.plot_nominal = nominal_plot
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
        return np.array([self.X[3,0]*np.cos(self.X[2,0]),
                         self.X[3,0]*np.sin(self.X[2,0]),
                         0,
                       0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [ 0, 0 ],
                          [ 0, 0 ],
                          [ 0, 1 ],
                          [ 1, 0 ],
                           ])  
        
    def f_nominal(self):
        return np.array([self.X_nominal[3,0]*np.cos(self.X_nominal[2,0]),
                         self.X_nominal[3,0]*np.sin(self.X_nominal[2,0]),
                         0,
                       0]).reshape(-1,1)
    
    def g_nominal(self):
        return np.array([ [ 0, 0 ],
                          [ 0, 0 ],
                          [ 0, 1 ],
                          [ 1, 0 ],
                           ])  
        
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
        
    def lyapunov(self, G):
        
        V = np.linalg.norm( self.X[0:2] - G[0:2] )**2
        dV_dx = np.append( 2*(self.X[0:2]-G[0:2]).T , [[0,0]] , axis=1       )
        
        return V, dV_dx
    
    # def nominal_input(self,G, type, d_min = 0.3):
    #     G = np.copy(G.reshape(-1,1))
    #     k_omega = 2.0 #0.5#2.5
    #     k_v = 3.0 #2.0 #0.5
    #     distance = max(np.linalg.norm( self.X[0:2,0]-G[0:2,0] ) - d_min,0)
    #     theta_d = np.arctan2(G[1,0]-self.X[1,0],G[0,0]-self.X[0,0])
    #     error_theta = wrap_angle( theta_d - self.X[2,0] )

    #     omega = k_omega*error_theta   
    #     v = k_v*( distance )*np.cos( error_theta )
    #     return np.array([v, omega]).reshape(-1,1)
    
    def nominal_input(self,target):
        # phi = self.X[2,0]
        # ui =  self.X[3,0]
        # los = self.X[0:2] - target[0:2]
        # los = los / np.linalg.norm(los)
        # los_angle = np.arctan2( self.X[1,0]-target[1,0], self.X[0,0]-target[0,0] )
        # error_angle = los_angle - phi
        # if error_angle > np.pi:
        #     error_angle = error_angle - 2*np.pi
        # if error_angle < -np.pi:
        #     error_angle = error_angle + 2*np.pi
        # tau_r = -5.0 * ( error_angle )
        # tau_u = 5.0 * ( 0.5 - ui ) * np.sign( -los[0,0]*np.cos(phi) -los[1,0]*np.sin(phi) )
        # return np.array([ tau_u, tau_r ]).reshape(-1,1)
    
        x = self.X[0,0]
        y = self.X[1,0]
        theta = self.X[2,0]
        v = self.X[3,0]
        
        k_omega = 2.0 #0.5#2.5
        k_v = 3.0 #2.0 #0.5
        
        theta_d = np.arctan2(target[1,0]-self.X[1,0],target[0,0]-self.X[0,0])
        error_theta = wrap_angle( theta_d - self.X[2,0] )
        omega = k_omega*error_theta  
        # print(d_min)
        distance = max(np.linalg.norm( self.X[0:2,0]-target[0:2,0] ) - d_min,0)
        speed = k_v*( distance )*np.cos( error_theta )
        u_r = k_v * ( speed - v )
        
        return np.array([u_r, omega]).reshape(-1,1)
        
    
        
        
       
    def agent_barrier(self, agent, d_min):
        
        f = np.array([self.X[3,0]*np.cos(self.X[2,0]),
                         self.X[3,0]*np.sin(self.X[2,0]),
                         0,
                       0]).reshape(-1,1)
        
        # 4 x 4
        Df_dx = np.array([ 
                         [ 0, 0, -self.X[3,0]*np.sin(self.X[2,0]), np.cos(self.X[2,0]) ],
                         [ 0, 0,  self.X[3,0]*np.cos(self.X[2,0]), np.sin(self.X[2,0]) ],
                         [ 0, 0, 0, 0],
                         [ 0, 0, 0, 0]
                           ])
        
        h = np.linalg.norm( self.X[0:2] - agent.X[0:2] )**2 - d_min**2           
        dh_dx = np.append( 2*( self.X[0:2] - agent.X[0:2] ).T, [[ 0, 0]], axis=1 )
                        
        # single derivative
        x_dot = f[0,0] # u cos(theta)
        y_dot = f[1,0] # u sin(theta)
        dx_dot_dx = Df_dx[0,:].reshape(1,-1)
        dy_dot_dx = Df_dx[1,:].reshape(1,-1)
        
        h_dot = 2*(self.X[0,0]-agent.X[0,0])*x_dot + 2*(self.X[1,0]-agent.X[1,0])*y_dot
        dh_dot_dx = 2*(self.X[0,0]-agent.X[0,0])*dx_dot_dx + 2*x_dot*np.array([[1,0,0,0]]) + 2*(self.X[2,0]-agent.X[1,0])*dy_dot_dx + 2*y_dot*np.array([[0,1,0,0]])
        
        h2 = h_dot + self.alpha1 * h
        dh2_dx = dh_dot_dx + self.alpha1 * dh_dx
        
       
            
        # assert(h2>=0)
        
        return h2, dh2_dx


if 0:
    plt.ion()
    fig = plt.figure()
    ax = plt.axes( xlim = (-2,2), ylim = (-2,2) )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect(1)
    
    dt = 0.01
    tf = 20
    num_steps = int(tf/dt)
    alpha1 = 2.0
    alpha2 = 7.0
    robot = Bicycle_2d( np.array([-0.5,-1.5,0,0.1]).reshape(-1,1), dt, ax, alpha1 = alpha1 )
    # 1,3,5 ok.. from top
    # 2,3,10 better
    obsX = [0.5, 0.5]
    obs2X = [1.5, 1.9]
    targetX = np.array([0.9, 0.9]).reshape(-1,1)
    d_min = 0.3
    obs1 = circle2D(obsX[0], obsX[1], d_min, ax, 0)
    obs2 = circle2D(obs2X[0], obs2X[1], d_min, ax, 0)
    
    ax.scatter(targetX[0,0], targetX[1,0],c='g')
    
    num_constraints = 2
    u = cp.Variable((2,1))
    u_ref = cp.Parameter((2,1), value = np.zeros((2,1)))
    A1 = cp.Parameter((num_constraints,2), value = np.zeros((num_constraints,2)))
    b1 = cp.Parameter((num_constraints,1), value = np.zeros((num_constraints,1)))
    delta = cp.Variable((num_constraints,1), value = np.zeros((num_constraints,1)))
    objective = cp.Minimize( cp.sum_squares( u - u_ref ) + 10 * cp.sum_squares( delta[0,0] ) + 100000 * cp.sum_squares(delta[1:,:]) ) 
    # factor_matrix = np.zeros((num_constraints,1)); factor_matrix[0,0] = 1
    const = [A1 @ u + b1 + delta >= 0]
    const += [ cp.abs( u[0,0] ) <= 2.0 ] #5
    const += [ cp.abs( u[1,0] ) <= 2.0 ] # 2.0
    cbf_controller = cp.Problem( objective, const )
    
    for i in range(num_steps):
        
        h1, dh1_dx = robot.agent_barrier( obs1, d_min )
        h2, dh2_dx = robot.agent_barrier( obs2, d_min )
        V, dV_dx = robot.lyapunov( targetX )
        u_ref.value = robot.nominal_input( targetX )
        # print(f"V:{V}, dV_dx:{dV_dx}")
        A1.value[0,:] = 1.0*(-dV_dx @ robot.g())
        A1.value[1,:] = 1.0*(dh1_dx @ robot.g())
        b1.value[0,:] = 1.0*(-dV_dx @ robot.f() - 1.0 * V )
        b1.value[1,:] = 1.0*(dh1_dx @ robot.f() + alpha2 * h1)
        cbf_controller.solve()#solver=cp.GUROBI, reoptimize=True)
        
        if cbf_controller.status!='optimal':
            print("ERROR in QP")
        # print(f"A:{A1.value}, b:{b1.value}, ref:{u_ref.value}")
        # print(f"control input: {u.value.T}")
        robot.step(u.value)
        # print(f"ui:{robot.X[3,0]}, vi:{robot.X[4,0]}")
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        