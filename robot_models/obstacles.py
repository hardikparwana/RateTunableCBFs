import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

class rectangle:

    def __init__(self,x,y,z,width,height,ax,id):
        self.X = np.array([x,y,z]).reshape(-1,1)
        self.width = width
        self.height = height
        self.id = id
        self.type = 'rectangle'

        self.render(ax)

    def render(self,ax):

        rect = Rectangle((self.X[0],self.X[1]),self.width,self.height,linewidth = 1, edgecolor='k',facecolor='k')
        ax.add_patch(rect)


class circle:

    def __init__(self,x,y,z,radius,ax,id):
        self.X = np.array([x,y,z]).reshape(-1,1)
        self.radius = radius
        self.id = id
        self.type = 'circle'

        self.render(ax)

    def render(self,ax):
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
        x = self.radius * np.cos(u) * np.sin(v) + self.X[0]
        y = self.radius * np.sin(u) * np.sin(v) + self.X[1]
        z = self.radius * np.cos(v) + self.X[2]
        ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r)
        
    def Xdot(self):
        return np.array([0,0,0]).reshape(-1,1)
