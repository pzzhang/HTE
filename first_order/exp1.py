from Chambolle import Chambolle
import numpy as np
import matplotlib.pyplot as plt


# generate data
nn = [6, 5, 4]
D = len(nn)
# clean signal
u0_exact = 0.0
u1_exact = [np.zeros(nn[d]) for d in range(D)]
u1_exact[0] = np.array([0.015,0,0,0,0,0])
u0_exact += np.mean(u1_exact[0])
u1_exact[0] -= np.mean(u1_exact[0])
# add noise
Ns = 10000
mm = [np.array([0.19,0.2,0.2,0.2,0.2,0.01]), 0.2*np.ones(5), np.array([0.1, 0.4, 0.3, 0.2])]
m = mm[0]
u_exact = u0_exact + u1_exact[0]
m = m[:, np.newaxis]*mm[1]
u_exact = u_exact[:, np.newaxis]+u1_exact[1]
m = m[:, :, np.newaxis]*mm[2]
u_exact = u_exact[:, :, np.newaxis]+u1_exact[2]
m *= Ns
v_exact = np.random.normal(loc=0, scale=0.1, size=m.shape)/np.sqrt(m)

g = u_exact + v_exact

# creat an object
chambo = Chambolle(g, m)

# create the topology
# chambo.set_graph(0,'circular')
# chambo.set_w(w_type=4)
# chambo.set_w(w_type=4)

# compute
# Lambda = 10. is good
Lambda = 20.0
n_it = 100000
energy, u0, u1 = chambo.PDALG2(Lambda=Lambda, n_it=n_it)

# plot the result
fig = plt.figure()
for d in range(D):
    plt.subplot(D, 1, d+1)
    # plt.plot(range(nn[1]), g, '-', label='data')
    plt.plot(range(nn[d]), u1[d], 'k^-', linewidth=2.0, label='recover')
    plt.plot(range(nn[d]), chambo.lssol[chambo.nind1[d]:chambo.nind1[d+1]], 'bs-', label='LS')
    plt.plot(range(nn[d]), u1_exact[d], 'ro-', label='true')
    plt.legend(loc='best')
    plt.title("Dimension = %d, Lambda = %1.2e, n_it = %1.2e" % (d, Lambda, n_it))
plt.show()
# plt.savefig('figure/exp1/Lambda = %1.2e_n_it = %1.2e.png' % (Lambda, n_it))
# plt.close(fig)
