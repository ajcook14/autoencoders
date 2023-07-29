import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

marker_size = mpl.rcParams['lines.markersize'] ** 2

import gzip, pickle, argparse
from net import Net



parser = argparse.ArgumentParser()
parser.add_argument('file_name', metavar='YYYYMMDD_HHMMSS', type=str, nargs='+',
                    help='file name of the pickled parameters object')

args = parser.parse_args()

file_name = args.file_name[0]

f = gzip.open(f'../data/manifold/{file_name}', 'rb')

layers, manifold, plotted, example = pickle.load(f)

f.close()

plotted_list = []
for p in plotted:

    if p[0] == 0:

        plotted_list.append(p)

    elif p[0] == 1:

        plotted_list.append((p[0], p[1], (p[2][0],)))

    else:

        print('strange behaviour in plotted parameter indexing')

p1, p2, p3 = tuple(plotted_list)

#line_t = np.arange(-40., 40., 5.)
#line = np.stack([line_t, np.zeros_like(line_t), -5 * np.ones_like(line_t)])

line_t = np.arange(0., 1., .05)
a = np.array([0., 30., -40]).reshape(3, 1)
b = np.array([0., -40., 30]).reshape(3, 1)

line = line_t * b + (1 - line_t) * a

def euclidean(x, y):

    return(np.sqrt(np.sum((x.flatten() - y.flatten())**2)))

for n in range(line_t.shape[0]):

    #n = 11
    min_dist = euclidean(manifold[0,:], line[:,n])
    min_index = 0
    for i in range(manifold.shape[0]):

        dist = euclidean(manifold[i,:], line[:,n])
        if dist < min_dist:

            min_dist = dist
            min_index = i

    for i in range(3):

        p = plotted[i]

        example[p[0]][p[1]][p[2]] = manifold[min_index, i]

    net = Net(layers, parameters=(example[0], example[1]))

    x = np.arange(0.0, 1.0, 0.01)#np.linspace(-0.5, 1.5, 50)
    y = np.zeros_like(x)
    for i in range(x.shape[0]):

        y[i] = net.feedforward(np.array([x[i]]))[0] - x[i]


    fig = plt.figure(figsize=(16, 8))#plt.figaspect(0.5))

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x, y, x, np.zeros_like(x))
    ax.set_title('$f(x) - x$ given the parameters on the right')
    plt.ylim(-0.2, 0.2)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(manifold[:,0], manifold[:,1], manifold[:,2], s=marker_size/2)
    ax.scatter(manifold[min_index,0], manifold[min_index,1], manifold[min_index,2], s=marker_size*2, c='r')
    ax.scatter(line[0,:], line[1,:], line[2,:], c='g')
    ax.scatter(line[0,n], line[1,n], line[2,n], s=marker_size*2)

    ax.set_title(f'bifurcation manifold for layers = {layers}')
    ax.set_xlabel('$%s_{%s}^{[%s]}$ (x)'%('w' if p1[0]==0 else 'b', ''.join(list(map(str, p1[2]))), p1[1]))
    ax.set_ylabel('$%s_{%s}^{[%s]}$ (y)'%('w' if p2[0]==0 else 'b', ''.join(list(map(str, p2[2]))), p2[1]))
    ax.set_zlabel('$%s_{%s}^{[%s]}$ (z)'%('w' if p3[0]==0 else 'b', ''.join(list(map(str, p3[2]))), p3[1]))

    plt.savefig(f'./outputs/manifold/{n}.png')
