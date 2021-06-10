"""This experiment tests for parameters that potentially yield a bifurcation in the zeros of the second derivative of the neural network of architecture 1-1-1, with sigmoid activations. The substitution y = sigmoid(a*x + b) has been made.
Last modified: 09/06/2021
Author: Andrew Cook
"""



from pyibex import *
from queue import Queue
import time

sigmoid = Function("z", "1/(1+exp(-z))")

f = Function("c", "d", "y", "2*y - 1 - c*y*(1 - y)*(1 - 2/(1 + exp(-d - c*y)))")

df_dy = Function("c", "d", "y", "2 - (c - 2*c*y)*(1 - 2/(1 + exp(-d - c*y))) + 2*c*(c*y - c*y^2)*(1/(1 + exp(-d - c*y)))*(1 - 1/(1 + exp(-d - c*y)))")

F = Function("c", "d", "y", "(2*y - 1 - c*y*(1 - y)*(1 - 2/(1 + exp(-d - c*y))), 2 - (c - 2*c*y)*(1 - 2/(1 + exp(-d - c*y))) + 2*c*(c*y - c*y^2)*(1/(1 + exp(-d - c*y)))*(1 - 1/(1 + exp(-d - c*y))))")

R = 30
x = IntervalVector([[-R, R], [-R, R], [0, 1]])



def arg_max(x):

    maximum = x[0]

    index = 0

    for i in range(len(x)):

        item = x[i]

        if item > maximum:

            maximum = item

            index = i

    return(index)



queue = Queue()
queue.append(x)

tol = 1e-2

verified = []

iteration = 0

while not queue.is_empty():

    print('%07d\r'%(len(queue),),end='')

    x = queue.serve()

    if not F.eval_vector(x).contains((0, 0)):

        continue

    dim = arg_max(x.diam())

    split = x.bisect(dim)

    for i in range(2):

        if split[i].max_diam() < tol:

            verified.append(split[i])

        else:

            queue.append(split[i])

    iteration += 1

print('')
print(iteration)
print(len(verified))
