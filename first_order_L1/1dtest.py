from Chambolle import Chambolle
import numpy as np
import matplotlib.pyplot as plt


# generate data
n = 24
# clean signal
u_exact = np.zeros(n)
u_exact[10:15] = 1
# add noise
mm = 100*np.ones(n)
v_exact = map(lambda m: np.random.normal(0, 1./np.sqrt(m)), mm)
g = u_exact + v_exact

# creat an object
chambo = Chambolle(g, mm)

# create the topology
A = np.zeros((n, n), dtype=bool)
A[range(1,n),range(0,n-1)] = True
A[range(0,n-1),range(1,n)] = True
chambo.set_graph([A])

# compute
Lambda = 0.1
n_it = 10000
energy, u0, u1 = chambo.PDALG2(Lambda=Lambda, n_it=n_it)

# plot the result
fig = plt.figure()
plt.plot(range(n), g, '-', label='data')
plt.plot(range(n), u0+u1[0], 'k*-', label='recover')
plt.plot(range(n), u_exact, 'ro-', label='true')
plt.legend(loc='best')
plt.title("TV recovery: Lambda = %1.2e, n_it = %1.2e" % (Lambda,n_it))
plt.xlabel('time')
# plt.show()
plt.savefig('figure/1dtest/Lambda = %1.2e_n_it = %1.2e.png' % (Lambda,n_it))
plt.close(fig)
