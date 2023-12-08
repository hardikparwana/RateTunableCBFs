import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt


xc = 2
yc = 2
R = 1
omega = 1

dt = 0.01
alpha = [1]
alpha_dot = []
tf = 1*np.pi/omega
T = int(tf/dt)

t = 0
h = []
x = []
y = []
xdot = []
ydot = []
hdot = []
ts = [0]

for i in range(T):
    
    
    x.append( xc + R * np.cos(omega*t) )
    y.append( yc + R * np.sin(omega*t) )
    xdot.append( - R * omega * np.sin(omega*t) )
    ydot.append(   R * omega * np.cos(omega*t) )
    
    h.append( x[-1]**2 + y[-1]**2 )
    hdot.append( 2*x[-1]*xdot[-1] + 2*y[-1]*ydot[-1] )
    
    alpha_dot.append( hdot[-1] + alpha[-1]*h[-1] - alpha[0]*h[0]*1.305 )
    alpha.append( alpha[-1]+alpha_dot[-1]*dt )
    ts.append(t+dt)
    
    t = t + dt
    
fig = plt.figure()
ax = plt.axes()
ax.plot(x,y,'r')

fig2, ax2 = plt.subplots(4,1)
ax2[0].plot(ts, alpha)
ax2[1].plot(ts[1:], alpha_dot)
ax2[2].plot(ts[1:], h)
ax2[3].plot(ts[1:], hdot)

ax2[0].set_ylabel('alpha')
ax2[1].set_ylabel('alpha_dot')
ax2[2].set_ylabel('h')
ax2[3].set_ylabel('hdot')

plt.show()