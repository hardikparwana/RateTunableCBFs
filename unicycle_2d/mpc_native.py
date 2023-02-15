import numpy as np
import cyipopt
import matplotlib.pyplot as plt
import time

dt = 0.05
N = 100
n = 3
m = 2
# starting point
x0 = np.zeros((N-1)*(n+m)+n)
X_init = np.array([-0.5,-0.5,np.pi/2])
obsX = np.array([0.7,0.7])
obsX2 = np.array([1.5,1.9])
d_obs = 0.3
goalX = np.array([2.0,2.0])

def step(x,u):
    return np.array( [ u[0]*np.cos(x[2]),
                      u[0]*np.sin(x[2]),
                      u[1] 
                    ] )*dt
    
def step_grad(x,u):
    
    Dstep_dx = np.array([ 
                         [ 0, 0, -u[0] * np.sin(x[2]) ],
                         [ 0, 0,  u[0] * np.cos(x[2]) ],
                         [ 0, 0, 0 ] 
                        ]) * dt
    
    Dstep_du = np.array([
                        [np.cos(x[2]), 0],
                        [np.sin(x[2]), 0],
                        [0, 1]
    ]) * dt
    
    return Dstep_dx, Dstep_du

class mpc():
    # x size: (N-1)*(n+m)+n
    def objective(self,x):
        """Returns the scalar value of the objective given x."""
        obj = 0
        for i in range(N):
            xi = x[(n+m)*i:(n+m)*i+n]
            ui = x[(n+m)*i+n:(n+m)*i+n+m]
            obj = obj + np.sum( 100*np.square(xi[0:2]-goalX) ) + np.sum( np.square(ui) )
        return obj
    
    def gradient(self,x):
        """Returns the gradient of the objective with respect to x."""
        i = 0
        xi = x[(n+m)*i:(n+m)*i+n]
        ui = x[(n+m)*i+n:(n+m)*i+n+m]
        grad = 2 * 10 * np.append((xi[0:2]-obsX),[0])
        grad = np.append( grad, ui  )
        for i in range(1,N):
            xi = x[(n+m)*i:(n+m)*i+n]
            ui = x[(n+m)*i+n:(n+m)*i+n+m]
            grad = np.append( grad, 2 * 10 * np.append((xi[0:2]-goalX),[0]) )
            grad = np.append( grad, 2 * ui  )
        return grad
    
    def constraints(self,x):
        """Returns the constraints. equality as well as inequality """ 
        
        ######################################### Equality constraints: 
        
        # Dynamics: (N-1)*n
        i = 0
        xi = x[(n+m)*i:(n+m)*i+n]
        cons = xi - X_init  # Initial state constraint
        for i in range(0,N-1):
            xi = x[(n+m)*i:(n+m)*i+n]
            ui = x[(n+m)*i+n:(n+m)*i+n+m]
            xi_1 = x[(n+m)*(i+1):(n+m)*(i+1)+n]        
            cons = np.append( cons, xi_1 - xi - step( xi, ui ) ) # cons[n*i:n*i+n] = xi_1 - xi - step( xi, ui )
            
        ############################### Inequality constraints
        
        # input bounds (N-1)*m
        for i in range(N-1):
            ui = x[(n+m)*i+n:(n+m)*i+n+m]
            cons = np.append( cons, 5 - ui[0]**2 )
            cons = np.append( cons, 5 - ui[1]**2 )
            
        # state constraints: obstacle at obsX
        for i in range(N): # 2*N constraints
            xi = x[(n+m)*i:(n+m)*i+n]
            cons = np.append( cons, (xi[0]-obsX[0])**2 + (xi[1]-obsX[1])**2 - d_obs**2 )         
            cons = np.append( cons, (xi[0]-obsX2[0])**2 + (xi[1]-obsX2[1])**2 - d_obs**2 )    
            
        return cons
        
    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        
        ######################## Equality 
        
        # Initial state
        num_states =  (N-1)*(n+m)+n
        jacobian0 = np.zeros( (n, num_states) )
        i = 0
        xi = x[(n+m)*i:(n+m)*i+n]
        jacobian0[0:n,0:n] = np.eye(n)
        
        # Dynamics: (N-1)*n x (N-1)*(n+m)+n
        jacobian1 = np.zeros( ((N-1)*n, num_states) )
        
        for i in range(0,N-1): # also the constraint index. size of each constraint: 3
            xi = x[(n+m)*i:(n+m)*i+n]
            ui = x[(n+m)*i+n:(n+m)*i+n+m]
            xi_1 = x[(n+m)*(i+1):(n+m)*(i+1)+n]               
            
            dx, du = step_grad(xi, ui)
            jacobian1[  i*n:i*n+n, i*(n+m):i*(n+m)+n,  ]   = - np.eye(n) - dx
            jacobian1[  i*n:i*n+n, i*(n+m)+n:i*(n+m)+n+m ] = - du
            jacobian1[  i*n:i*n+n, (i+1)*(n+m):(i+1)*(n+m)+n ] = np.eye(n)
            
        #################### Inequality
        
        # input bounds: (N-1)*m
        jacobian2 = np.zeros( ((N-1)*m, num_states) )
        for i in range(N-1):
            ui = x[(n+m)*i+n:(n+m)*i+n+m]
            jacobian2[ i*m:i*m+m, i*(n+m)+n:i*(n+m)+n+m ] = np.diag( -2*ui )
            
        # state constraints: obstacle at obsX: N constraints
        jacobian3 = np.zeros( (2*N, num_states) )
        for i in range(N): # N constraints
            xi = x[(n+m)*i:(n+m)*i+n]
            jacobian3[ 2*i, (n+m)*i:(n+m)*i+n ] = np.append( 2*( xi[0:2]-obsX[0:2] ), [0] )  
            jacobian3[ 2*i+1, (n+m)*i:(n+m)*i+n ] = np.append( 2*( xi[0:2]-obsX2[0:2] ), [0] )  
            
        ##################################################
        # Append all
        jacobian = np.append( jacobian0, np.append( jacobian1, np.append(jacobian2, jacobian3, axis=0), axis=0 ), axis=0 )
        
        return jacobian

 
            
            
prob = mpc()

# Constraints
lb = -100 * np.ones((N-1)*(n+m)+n)
ub =  100 * np.ones((N-1)*(n+m)+n)

equality = np.zeros( N*n )
inequality_lower = np.zeros( (N-1)*m + N )
inequality_upper = 2.0e19 * np.ones( (N-1)*m + N )

cl = np.append(equality, inequality_lower) 
cu = np.append(equality, inequality_upper)

x0 = np.zeros( (N-1)*(n+m)+n )
x0[0:3] = X_init

prob.gradient(x0)

nlp = cyipopt.Problem(
   n=len(x0),
   m=len(cl),
   problem_obj=mpc(),
   lb=lb,
   ub=ub,
   cl=cl,
   cu=cu,
)

nlp.add_option('mu_strategy', 'adaptive')
nlp.add_option('tol', 1e-7)
nlp.add_option('linear_solver', 'ma57')
t0 = time.time()
X_sol, info = nlp.solve(x0)
print("time: ", time.time()-t0)
# print(X_sol)

# Plot the system
Xs = np.zeros((3,1))
Us = np.zeros((2,1))
for i in range(N-1):
    xi = X_sol[(n+m)*i:(n+m)*i+n]
    ui = X_sol[(n+m)*i+n:(n+m)*i+n+m]
    Xs = np.append( Xs, xi.reshape(-1,1), axis=1 )
    Us = np.append( Us, ui.reshape(-1,1), axis=1 )
    # ui = x[(n+m)*i+n:(n+m)*i+n+m]
    
fig, ax = plt.subplots(1,1)
plot_x_lim = (-0.5,2.5)  
plot_y_lim = (-0.5,2.5) 
ax.set_xlim( plot_x_lim )
ax.set_ylim( plot_y_lim )
circ = plt.Circle((obsX[0],obsX[1]),d_obs, linewidth = 1, edgecolor='k',facecolor='k')
ax.add_patch(circ)
circ2 = plt.Circle((obsX2[0],obsX2[1]),d_obs, linewidth = 1, edgecolor='k',facecolor='k')
ax.add_patch(circ2)
ax.plot(Xs[0,1:], Xs[1,1:],'r')

# plt.figure()
plt.show()

