def lorenz(dt,sigma=10.0,beta=2.66667,ro=28.):
 def l(x,y,z):
  xn = y*dt*sigma + x*(1 - dt*sigma)
  yn = x*dt*(ro-z) + y*(1-dt)
  zn = x*y*dt + z*(1 - dt*beta)
  return (xn,yn,zn)
 return l

def trajectory(system,origin,steps):
 t = [origin]
 for _ in range(steps):
  t.append(system(*t[-1]))
 return t

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot(trajectory):
 fig = plt.figure()
 ax = fig.gca(projection='3d')
 ax.plot([x for (x,_,_) in trajectory],
  [y for (_,y,_) in trajectory],
  [z for (_,_,z) in trajectory])
 plt.show()

steps = 100000
dt = 1e-3

l = lorenz(dt)
t = trajectory(l,(1.0,0.0,0.0),steps)
plot(t)