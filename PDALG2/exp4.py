from Chambolle import Chambolle
import numpy as np
import matplotlib.pyplot as plt


# generate data
nn = [10, 3, 5, 4]
D = len(nn)
# clean signal
u0_exact = 0.05
u1_exact = [np.zeros(nn[d]) for d in range(D)]
u1_exact[0][0:2] = np.array([0.03,-0.03])
u0_exact += np.mean(u1_exact[0])
u1_exact[0] -= np.mean(u1_exact[0])
u1_exact[1][1] = -0.02
u0_exact += np.mean(u1_exact[1])
u1_exact[1] -= np.mean(u1_exact[1])
# add noise
Ns = 10000
mm = [0.1*np.ones(10), np.array([0.2, 0.5, 0.3]), 0.2*np.ones(5), np.array([0.1, 0.4, 0.3, 0.2])]
m = mm[0]
u_exact = u0_exact + u1_exact[0]
m = m[:, np.newaxis]*mm[1]
u_exact = u_exact[:, np.newaxis]+u1_exact[1]
m = m[:, :, np.newaxis]*mm[2]
u_exact = u_exact[:, :, np.newaxis]+u1_exact[2]
m = m[:, :, :, np.newaxis]*mm[3]
u_exact = u_exact[:, :, :, np.newaxis]+u1_exact[3]
m *= Ns
v_exact = np.random.normal(loc=0, scale=0.1, size=m.shape)/np.sqrt(m)

g = u_exact + v_exact

# creat an object
chambo = Chambolle(g, m)

# create the topology
# A = [None]*D
# for d in range(D):
#     A[d] = np.zeros((nn[d], nn[d]), dtype=bool)
#     A[d][range(1,nn[d]),range(0,nn[d]-1)] = True
#     A[d][range(0,nn[d]-1),range(1,nn[d])] = True
# d = 1
# A[d] = np.ones((nn[d], nn[d]), dtype=bool)
# A[d][range(0,nn[d]),range(0,nn[d])] = False
# chambo.set_graph(A)
chambo.set_w(w_type=1)

# compute
# Lambda = 10. is good
Lambda = 250.0
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
# plt.savefig('figure/exp4/Lambda = %1.2e_n_it = %1.2e.png' % (Lambda, n_it))
# plt.close(fig)
