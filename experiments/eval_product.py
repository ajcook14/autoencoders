from pyibex import *
from tubex_lib import *

sig = lambda z: f"(1/(1 + exp(-{z})))"

sigmoid = Function("z[2]", f"({sig('z[0]')},{sig('z[1]')})")

dsig = lambda z: f"({sig(z)}*(1-{sig(z)}))"

f = Function("y[2]", f"( (y[0]-y[1])/({sig('y[0]')}-{sig('y[1]')}) )^2 * {dsig('y[0]')} * {dsig('y[1]')}")
f = Function("y[2]", f"( ((y[0]-y[1])/sqrt(2)-(y[0]y[1])/({sig('y[0]')}-{sig('y[1]')}) )^2 * {dsig('y[0]')} * {dsig('y[1]')}")

r = 1e100
x = IntervalVector([[-r,-1],[1,r]])
print(x)

print(f.eval_vector(x))
