import gzip, pickle, argparse
import numpy as np
from parameters import Parameters
from net import Net

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

marker_size = mpl.rcParams['lines.markersize'] ** 2



parser = argparse.ArgumentParser()
parser.add_argument('file_name', metavar='YYYYMMDD_HHMMSS', type=str, nargs='+',
                    help='file name of the pickled parameters object')

args = parser.parse_args()

file_name = args.file_name[0]

f = gzip.open(f'./data/quadratic/{file_name}', 'rb')
params = pickle.load(f)
f.close()

layers = params.layers
encoder_params = (params.weights[:2], params.biases[:2])
encoder_layers = layers[:2]
encoder = Net(encoder_layers, parameters=encoder_params)

decoder_layers = layers[2:]
decoder_params = (params.weights[2:], params.biases[2:])
decoder = Net(decoder_layers, parameters=decoder_params)

# compute encoder output
m = 20

r = 0
xi = np.arange(-r, r + 1, (2*r + 1)/m)
yi = np.arange(-r, r + 1, (2*r + 1)/m)

xx, yy = np.meshgrid(xi, yi)

zz = np.zeros((m, m))

for i in range(m):

    for j in range(m):

        output = encoder.feedforward( np.array([xx[i, j], yy[i, j]]) )

        zz[i, j] = output[0]

# compute decoder output
xh = np.arange(0.0, 1.0, 1/m)
z = np.zeros((2, m))

for i in range(m):

    output = decoder.feedforward( np.array([xh[i]]) )

    z[:,i] = output

# plot the encoder and decoder
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

pos = ax[0].imshow(zz, interpolation='bilinear', cmap=cm.RdYlGn, origin='lower', extent=[-r, r + 1, -r, r + 1], vmax=1.0, vmin=0.0)

fig.colorbar(pos, ax=ax[0])

ax[1].scatter(z[0,:], z[1,:], s=marker_size/4, c='b')
ax[1].set_xlim([0, 1])
ax[1].set_ylim([0, 1])

plt.show()
#plt.savefig(f'./figures/quadratic/enc-dec/{file_name}.png')
