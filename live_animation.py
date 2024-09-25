import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(0,5,5000)
y = np.ones(x.shape)
z = 5* np.ones(x.shape)
p  = ax.plot(x[0],y[0],z[0], 'r')#,c = 'b', s = 20)
ps = ax.scatter( x[0], y[0], z[0], s=60 )
T = x.shape[0]
metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
name = 'test_animated_plot.mp4'
writer = FFMpegWriter(fps=12, metadata=metadata)
with writer.saving(fig, name, 100): # this is only needed to record into a video
    for t in range(T):

        #scatter
        ps._offsets3d = ( [x[t]],[y[t]], [z[t]] )

        # plot
        p[0].set_xdata(x[0:t+1])
        p[0].set_ydata(y[0:t+1])
        p[0].set_3d_properties(z[0:t+1])

        writer.grab_frame() # thius is only needed to record into a video
        fig.canvas.draw()
        fig.canvas.flush_events()