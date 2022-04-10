import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
r = 1

for i in range(int(1e6)):

    r = np.tanh(r)

r = 1 / r

y = np.tanh(x / r)

for i in range(int(1e6)):

    y = np.tanh(y)

y = r * y

fig, ax = plt.subplots()

z = np.tanh(x)
ax.plot(x, z, label='raw')
ax.plot(x, y, label='iterated')
ax.legend()

plt.show()
