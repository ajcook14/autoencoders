from interval_bisection import *

def f(x):

    return(0.5 * x + 0.5 - x)

u = Interval(0, 2)
v = Interval(0, 2)
x = np.array([u, v])
tol = 0.1
queue = Queue()
queue.append(x)

result = interval_bisection(f, queue, tol)
print(len(result))

fig, ax = plt.subplots()

ax.set_xlim([0, 2])
ax.set_ylim([0, 2])

rectangles(ax, result)
#rect = Rectangle((0.1, 0.1), w, h, linewidth=2, edgecolor='r', fill=True, clip_on=False)

plt.show()
