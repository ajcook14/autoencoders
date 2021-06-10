import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

marker_size = mpl.rcParams['lines.markersize'] ** 2

def sigmoid(z):

    return( 1 / (1 + np.exp(-z)) )

def rotate(u, theta):

    a = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return(np.dot(a, u))

def animate(i):

    rot_v = rotate(vv, 2*np.pi*i)
    rot_h = rotate(hh, 2*np.pi*i)

    rot_v = sigmoid(rot_v)
    rot_h = sigmoid(rot_h)

    points_v.set_offsets(rot_v.T)
    points_h.set_offsets(rot_h.T)

n = 40 # number of lines
m = 200 # points per line

x = np.linspace(-10, 10, n)
y = np.linspace(-10, 10, m)

xx, yy = np.meshgrid(x, y)
vv = np.array([xx.flatten(), yy.flatten()])
hh = np.array([yy.flatten(), xx.flatten()])

data_v = sigmoid(vv)
data_h = sigmoid(hh)

fig, ax = plt.subplots(figsize=(12, 12))
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

#ax.scatter(vv[0,:], vv[1,:], s=marker_size/4, c='c')
points_v = ax.scatter(data_v[0,:], data_v[1,:], s=marker_size/4, c='b')
#ax.scatter(hh[0,:], hh[1,:], s=marker_size/4, c='m')
points_h = ax.scatter(data_h[0,:], data_h[1,:], s=marker_size/4, c='r')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)

interval = 365
ani = animation.FuncAnimation(fig, animate, np.arange(0, 1, 1/interval), interval=interval, blit=False)

ani.save('sigmoid.mp4', writer=writer)
