from pyibex import *
from tubex_lib import *
from queue import Queue
from copy import deepcopy

def argmax(a):

    index = 0

    for i in range(1, len(a)):

        if a[i] > a[index]:

            index = i

    return(index)

sig = lambda z: f"(1/(1 + exp(-{z})))"

sigmoid = Function("z[2]", f"({sig('z[0]')},{sig('z[1]')})")

dsig = lambda z: f"({sig(z)}*(1-{sig(z)}))"

f = Function("y[2]", f"( (y[0]-y[1]) / ({sig('y[0]')} - {sig('y[1]')}) )^2 * {dsig('y[0]')} * {dsig('y[1]')}")

y = IntervalVector([[-oo,oo],[-oo,oo]])


queue = Queue()
queue.append(deepcopy(y))

verified = []

gt_one = Interval(1,oo)

beginDrawing()
fig = VIBesFigMap("Map")
fig.set_properties(50, 50, 400, 400)
fig.axis_limits(IntervalVector([[0,1],[0,1]]))

while not queue.is_empty():

    y = queue.serve()

    fig.draw_box(sigmoid.eval_vector(y), "red")

    #y.inflate(1e-10)

    maxdim = argmax(y.diam())

    if y.max_diam() < 1e-1 or not y[maxdim].is_bisectable():

        verified.append(deepcopy(y))

        continue

    if f.eval(y).overlaps(gt_one):

        split = y.bisect(maxdim)

        queue.append(deepcopy(split[0]))
        queue.append(deepcopy(split[1]))



fig.show()

endDrawing()



