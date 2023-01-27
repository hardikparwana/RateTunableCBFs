from jax.config import config
# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update('jax_platform_name', 'cpu')

# for automatic differentiation of functions
import jax.numpy as np
from jax import jit, grad, jacfwd, jacrev

from cyipopt import minimize_ipopt
import matplotlib.pyplot as plt

# Variables
# x: n x N = Nn
# u: m x N = mN

dt = 0.1
N = 30
n = 3
m = 2
# starting point
x0 = np.zeros((N-1)*(n+m)+n)
X_init = np.array([0,0,np.pi/2])
obsX = np.array([0.7,0.7])
d_obs = 0.2
goalX = np.array([1.5,1.5])

def step(x,u):
    return np.array( [ u[0]*np.cos(x[2]),
                      u[0]*np.sin(x[2]),
                      u[1] 
                    ] )*dt
    
def objective(x):
    obj = 0
    for i in range(N-1):
        xi = x[(n+m)*i:(n+m)*i+n]
        ui = x[(n+m)*i+n:(n+m)*i+n+m]
        obj = obj + np.sum( 10*np.square(xi[0:2]-obsX) ) + np.sum( np.square(ui) )
    return obj

def eq_constraints(x):
    
    # number of constraints
    # dynamics constraints: (N-1)*n
    i = 0
    xi = x[(n+m)*i:(n+m)*i+n]
    cons = xi - X_init
    for i in range(0,N-1):
        xi = x[(n+m)*i:(n+m)*i+n]
        ui = x[(n+m)*i+n:(n+m)*i+n+m]
        xi_1 = x[(n+m)*(i+1):(n+m)*(i+1)+n]        
        cons = np.append( cons, xi_1 - xi - step( xi, ui ) ) # cons[n*i:n*i+n] = xi_1 - xi - step( xi, ui )
    return cons

def ineq_constraints(x):
    # input bounds (N-1)*m
    i = 0
    ui = x[(n+m)*i+n:(n+m)*i+n+m]
    cons = 5 - ui[0]**2 
    cons = np.append( cons, 5 - ui[1]**2 )
    for i in range(1,N-1):
        ui = x[(n+m)*i+n:(n+m)*i+n+m]
        cons = np.append( cons, 5 - ui[0]**2 )
        cons = np.append( cons, 5 - ui[1]**2 )
        
    # state constraints: obstacle at obsX
    for i in range(N):
        xi = x[(n+m)*i:(n+m)*i+n]
        cons = np.append( cons, (xi[0]-obsX[0])**2 + (xi[1]-obsX[1])**2 - d_obs**2 )
    return cons

def constraints(x):
    return np.append( np.append( eq_constraints(x), -eq_constraints(x) ), ineq_constraints(x) )
        
# jit the functions
obj_jit = jit(objective)
con_eq_jit = jit(eq_constraints)
con_ineq_jit = jit(ineq_constraints)
con_jit = jit(constraints)

# build the derivatives and jit them

# Objective
obj_grad = jit(grad(obj_jit))  # objective gradient
obj_hess = jit(jacrev(jacfwd(obj_jit))) # objective hessian

# Equality constraints
con_eq_jac = jit(jacfwd(con_eq_jit))  # jacobian
con_eq_hess = jacrev(jacfwd(con_eq_jit)) # hessian
con_eq_hessvp = jit(lambda x, v: con_eq_hess(x) * v[0]) # hessian vector-product

# Inequality constraints
con_ineq_jac = jit(jacfwd(con_ineq_jit))  # jacobian
con_ineq_hess = jacrev(jacfwd(con_ineq_jit))  # hessian
con_ineq_hessvp = jit(lambda x, v: con_ineq_hess(x) * v[0]) # hessian vector-product

# all cons
con_jac = jit(jacfwd(con_jit))  # jacobian
con_hess = jacrev(jacfwd(con_jit)) # hessian
con_hessvp = jit(lambda x, v: con_hess(x) * v[0]) # hessian vector-product
        
    
# constraints
# cons = [
#     {'type': 'eq', 'fun': con_eq_jit, 'jac': con_eq_jac, 'hess': con_eq_hessvp},
#     {'type': 'ineq', 'fun': con_ineq_jit, 'jac': con_ineq_jac, 'hess': con_ineq_hessvp}
#  ]
cons = [
    {'type': 'ineq', 'fun': con_jit, 'jac': con_jac, 'hess': con_hessvp}
 ]

# variable bounds: 1 <= x[i] <= 5
bnds = [(-100, 100) for _ in range(x0.size)]

# executing the solver
res = minimize_ipopt(obj_jit, jac=obj_grad, hess=obj_hess, x0=x0, bounds=bnds,
                  constraints=cons, options={'disp': 5, 'maxiter': 6000,'linear_solver': 'ma57'})
    
X = res.x

# Plot the system
Xs = np.zeros((3,1))
Us = np.zeros((2,1))
for i in range(N-1):
    xi = X[(n+m)*i:(n+m)*i+n]
    ui = X[(n+m)*i+n:(n+m)*i+n+m]
    Xs = np.append( Xs, xi.reshape(-1,1), axis=1 )
    Us = np.append( Us, ui.reshape(-1,1), axis=1 )
    # ui = x[(n+m)*i+n:(n+m)*i+n+m]
    
fig, ax = plt.subplots(1,1)
circ = plt.Circle((obsX[0],obsX[1]),d_obs, linewidth = 1, edgecolor='k',facecolor='k')
ax.add_patch(circ)
ax.plot(Xs[0,1:], Xs[1,1:],'r')

# plt.figure()
plt.show()