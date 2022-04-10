import numpy as np
from numpy.polynomial import *
import matplotlib.pyplot as plt
from pyibex import Interval

rng = np.random.default_rng(2)

deg = 4

N = 4

id = Polynomial([0., 1.])

valid = lambda x: -1. < x and x < 1.



def bisection(f, lb, ub):

    if ub - lb < 1e-14:

        return( (lb + ub)/2 )

    mid = (lb + ub)/2

    if min( f(lb), f(mid) ) <= 0. <= max( f(lb), f(mid) ):

        return( bisection(f, lb, mid) )

    if min( f(ub), f(mid) ) <= 0. <= max( f(ub), f(mid) ):

        return( bisection(f, mid, ub) )

iteration = 0
while True:

    print('iteration = {}\r'.format(iteration), end='')

    free = rng.uniform(-3, 3, deg - 3)

    A = np.eye(deg + 1)

    A[0] = np.array(range(0, deg + 1))
    A[1] = A[0] * np.array([(-1)**i for i in range(0, deg + 1)])
    A[2] = np.ones(deg + 1)
    A[3] = np.array([(-1)**i for i in range(0, deg + 1)])

    b = np.array([0., 0., 1., -1.] + list(free))

    alpha = np.linalg.solve(A, b)

    p = Polynomial(alpha)

    extrema = list(map(valid,list(Polynomial(polynomial.polyder(p.coef, 1)).roots()))).count(True)
    inflection = list(map(valid,list(Polynomial(polynomial.polyder(p.coef, 2)).roots()))).count(True)

    if extrema == 0 and inflection == 1:

        while True:

            a = rng.uniform(-2,2,N)
            b = rng.uniform(-1,1,N)

            if not 0. in list(a):

                break

        f = Polynomial([b[0], a[0]])

        u = (Interval(-1, 1) - b[0])/a[0]

        j_lb = u.lb()

        j_ub = u.ub()

        for i in range(1, N):

            f = a[i] * p(f) + b[i]

            tup = ( f(j_lb), f(j_ub) )

            if tup[0] < tup[1]:

                sign = 1.

            else:

                sign = -1.

            v = Interval( min(tup), max(tup) )

            if not v.overlaps(Interval(-1, 1)):

                print('break')

                break

            if v.is_subset(Interval(-1, 1)):

                tup = (j_lb, j_ub)

            if v.lb() < -1.:

                # solve for f(x_1) = -1.
                x_1 = bisection(f + 1, j_lb, j_ub)

                if v.ub() > 1.: # vl -1 1 vu

                    # solve for f(x_2) = 1.
                    x_2 = bisection(f - 1, j_lb, j_ub)

                    tup = (x_1, x_2)

                else: # vl -1 vu 1

                    if sign > 0.:

                        tup = (x_1, j_ub)

                    else:

                        tup = (j_lb, x_1)

            elif v.ub() > 1.: # -1 vl 1 vu

                # solve for f(x_2) = 1.
                x_2 = bisection(f - 1, j_lb, j_ub)

                if sign > 0.:

                    tup = (j_lb, x_2)

                else:

                    tup = (x_2, j_ub)

            if tup[0] > tup[1]:

                tup = (tup[1], tup[0])

            j_lb, j_ub = tup


        f = p(f)

        mapsin = lambda x: x.imag == 0. and j_lb < x and x < j_ub

        zeroes = list(map(mapsin,list(f((f - id).roots())))).count(True)

        if True:#zeroes > 3:

            #print(zeroes)
            #print(f((f - id).roots()))
            #print(j_lb, j_ub)
            fig, ax = plt.subplots()

            x = np.linspace(j_lb, j_ub, 100)

            #print(f(x).max(), f(x).min())

            ax.plot(x, f(x))

            plt.show()

    iteration += 1
