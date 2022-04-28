import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import gzip, pickle, argparse



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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(manifold[:,0], manifold[:,1], manifold[:,2])
ax.set_title(f'bifurcation manifold for layers = {layers}')
ax.set_xlabel('$%s_{%s}^{[%s]}$'%('w' if p1[0]==0 else 'b', ''.join(list(map(str, p1[2]))), p1[1]))
ax.set_ylabel('$%s_{%s}^{[%s]}$'%('w' if p2[0]==0 else 'b', ''.join(list(map(str, p2[2]))), p2[1]))
ax.set_zlabel('$%s_{%s}^{[%s]}$'%('w' if p3[0]==0 else 'b', ''.join(list(map(str, p3[2]))), p3[1]))

plt.show()
