from pyibex import *
from tubex_lib import *
from queue import Queue

def argmax(a):

    index = 0

    for i in range(1, len(a)):

        if a[i] > a[index]:

            index = i

    return(index)

ctc_d = CtcFunction(Function("x", "a", "b", "d", "d/a - 1/(1 + exp(-a*x - b)) + 1/(1 + exp(-a*x - b))^2"))
ctc_x = CtcFunction(Function("x", "a", "b", "1/(1 + exp(-a*x - b)) - x"))

x = IntervalVector(0, 1)
a = IntervalVector(4, oo)
b = IntervalVector()
d = IntervalVector(1, oo)

cn = ContractorNetwork()
cn.add(ctc_d, [x, a, b, d])
cn.add(ctc_x, [x, a, b])

cn.contract()

x_ = IntervalVector([[0,1],[4,oo],[-oo,oo],[1,oo]])#cart_prod(x, a, b)

queue = Queue()
queue.append(x_)

verified = []

while not queue.is_empty():

    x_ = queue.serve()

    x_.inflate(1e-10)

    if x_.max_diam() < 1e-5 or not x_.is_bisectable():

        verified.append(x_)

        continue

    x, a, b, d = x_

    if x_.contains([0.,0.,0.]):

        maxdim = argmax(x_.diam())

        split = x_.bisect(maxdim)

        queue.append(split[0])
        queue.append(split[1])


