import numpy as np

class ACC2D:
    
    def __init__(self,X0,dt,ax,id = 0, color='r',palpha=1.0, plot=True):
        '''
        X0: initial state: position, velocity of ego + distance between ego and 
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'ACC2D'        
        
        X0 = X0.reshape(-1,1)
        self.X = X0
        self.dt = dt
        self.id = id
        self.color = color
        self.palpha = palpha

        self.U = np.array([0,0]).reshape(-1,1)
        self.xdot = np.zeros((3,1))
       
        # Plot handles
        self.plot = plot
        if self.plot:
            self.ax = ax
            self.body = ax.scatter([],[],c=color,alpha=palpha,s=30)
            self.render_plot()
        self.Xs = np.copy(self.X)
        self.Us = np.copy(self.U)
        
        self.m = 1650
        self.f0 = 0.1
        self.f1 = 5
        self.f2 = 0.25
        self.psc = 100
        self.gr = 9.81
        self.v0 = 27.7
        self.vd = 24
        self.a_max = 0.3 * self.gr
        self.af = 0.4#0.25
        self.vmin = 0.0
        self.vmax = 30 #30
        self.lp = 10
        
    def Fr(self):
        return self.f0 + self.f1 * self.X[0,0] + self.f2 * self.X[0,0]**2
        
    def f(self):
        Fr = self.f0 + self.f1 * self.X[0,0] + self.f2 * self.X[0,0]**2
        aL = 0
        return np.array([ -1/self.m * Fr, aL, self.X[1,0] - self.X[0,0]  ]).reshape(-1,1)
    
    def g(self):
        # return np.array([ [1/self.m], [0], [0] ])*self.m
        return np.array([ [1/self.m*self.m, 0], [0, 1], [0, 0] ])
        
    def step(self,U): #Just holonomic X,T acceleration

        self.U = U.reshape(-1,1)
        self.xdot = self.f() + self.g() @ self.U
        self.X = self.X + self.xdot * self.dt        
        # self.render_plot()
        self.Xs = np.append(self.Xs,self.X,axis=1)
        self.Us = np.append(self.Us,self.U,axis=1)
        return self.X

    def render_plot(self):
        if self.plot:
            x = np.array([self.X[0,0],self.X[1,0]])
            self.body.set_offsets([x[0],x[1]])
            
    def vel_barrier1(self):
        v_min = 0.0
        h = self.X[0,0] - self.vmin
        dh_dx = np.array([[1,0,0]])
        return h, dh_dx
    
    def vel_barrier2(self):
        v_max = 0.0
        h = self.vmax - self.X[0,0]
        dh_dx = np.array([[-1,0,0]])
        return h, dh_dx
            
    def lyapunov(self):
        V = (self.X[0,0] - self.vd)**2
        dV_dx = 2 * np.array([[ (self.X[0,0]-self.vd), 0, 0 ]])
        return V, dV_dx
    
    def distance_barrier(self):
        h = self.X[2,0] - self.lp
        h_dot = self.X[1,0] - self.X[0,0]
        dh_dot_dx = np.array([[-1, 1, 0]])
        return h, h_dot, dh_dot_dx