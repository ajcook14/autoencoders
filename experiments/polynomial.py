import numpy as np
from numpy.polynomial import *
import matplotlib.pyplot as plt
from pyibex import *

r = 2
x = np.linspace(-r, r, 100)
#p = lambda x: -(x**3)/2 + 3*x/2
p = Polynomial([-0.6, 1.07402, 1.2, 0.351954, -0.6, -0.425977])

I = Interval(-1, 1)
J = [I]

a = [1, -1]
b = [0, 0.25]
c = a[0]
d = b[0]

y = p(a[0] * x + b[0])

for i in range(1, len(a)):

    y = p(a[i] * y + b[i])
    c = a[i] * c
    d = a[i] * d + b[i]
    J.append( J[-1] & ((I - d)/c) )

fig, ax = plt.subplots()

ax.plot(x, y)
l = np.array([-r, r])
#for i in range(len(J)):
#    interval = J[i]
#    ax.plot(np.array([interval.lb(), interval.lb()]), l, c=(i/len(J), i/len(J), i/len(J)/2))
#    ax.plot(np.array([interval.ub(), interval.ub()]), l, c=(i/len(J), i/len(J), i/len(J)/2))
interval = J[-1]
print(interval)
ax.plot(np.array([interval.lb(), interval.lb()]), l, c='r')
ax.plot(np.array([interval.ub(), interval.ub()]), l, c='r')

ax.set_ylim(-r, r)
plt.show()
